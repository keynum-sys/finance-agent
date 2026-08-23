"""Chroma 向量库: 章节级入库 + 带页码溯源的检索问答。

设计要点:
- 入库单元是「子块」: 章节 chunk 按页边界 + 500 字窗口再切分,
  使 embedding 不超出 bge-small-zh 的 512 token 窗口, 同时把
  溯源粒度细化到页级(项目核心卖点)。
- 检索答案必须附带引用: [Citation(page, section, snippet)]
- embed_fn / chat_fn 均可注入, 单元测试全程离线。

embedding: BAAI/bge-small-zh-v1.5 (本地 ONNX, 512 维)。
bge 中文系列的查询侧需加指令前缀以对齐检索场景。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# 国内网络: huggingface.co 与 xet CDN 均不可达。
# 必须在 import chromadb(会连带导入 huggingface_hub)之前设置,
# 否则 huggingface_hub 读不到这两个环境变量。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import chromadb

from finance_agent.config import settings
from finance_agent.parsing.extractor import ChatFn
from finance_agent.parsing.pdf_parser import Chunk

EmbedFn = Callable[[list[str]], list[list[float]]]

# bge 中文 v1.5 官方推荐: 查询侧加检索指令前缀
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_WINDOW = 500  # 子块最大字符数(bge-small-zh 窗口约 512 token)


def _default_chat(messages: list[dict]) -> str:
    """RAG 问答的 LLM 调用: 自然语言输出, 不用 JSON mode(区别于抽取场景)。"""
    from openai import OpenAI

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content or ""


@dataclass
class Citation:
    page: int
    section: str
    snippet: str


_MODEL_REPO = "Qdrant/bge-small-zh-v1.5"  # fastembed 打包版 bge-small-zh 的 HF 仓库
_MIRROR_BASE = "https://hf-mirror.com/Qdrant/bge-small-zh-v1.5/resolve/main/"
_REQUIRED_FILES: dict[str, int] = {  # 文件名 -> 最小合法字节数
    "model_optimized.onnx": 10_000_000,  # 完整约 90MB
    "config.json": 1,
    "tokenizer.json": 1,
    "special_tokens_map.json": 1,
    "tokenizer_config.json": 1,
}


def _model_ok(model_dir: Path) -> bool:
    """所有必需文件存在且达到最小字节数才算完整。"""
    return all(
        (model_dir / name).exists() and (model_dir / name).stat().st_size >= min_size
        for name, min_size in _REQUIRED_FILES.items()
    )


def _download_file(url: str, dest: Path, min_size: int, retries: int = 3) -> None:
    """单文件下载: 先写临时文件, 校验大小后原子改名。

    不用 huggingface_hub: 其 HEAD/xet 机制在本项目验证的网络环境下
    会对小文件产生 0 字节损坏结果, urllib 直下反而稳定。
    """
    import time
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "finance-agent"})
            with (
                urllib.request.urlopen(req, timeout=120) as resp,
                open(tmp, "wb") as w,
            ):
                while chunk := resp.read(1 << 16):
                    w.write(chunk)
            if tmp.stat().st_size >= min_size:
                tmp.replace(dest)
                return
            raise RuntimeError(f"大小校验失败: {tmp.stat().st_size} < {min_size}")
        except Exception as e:  # noqa: BLE001 - 弱网重试
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载 {url} 失败(已重试 {retries} 次): {last_err}")


def _ensure_model() -> Path:
    """确保本地模型文件完整, 返回模型目录(首次运行下载, 之后走缓存)。"""
    model_dir = Path.home() / ".cache" / "fastembed" / "bge-small-zh-v1.5"
    if _model_ok(model_dir):
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, min_size in _REQUIRED_FILES.items():
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size >= min_size:
            continue  # 已有完整文件, 跳过(避免重复下载 90MB 模型)
        _download_file(_MIRROR_BASE + name, dest, min_size)
    if not _model_ok(model_dir):
        raise RuntimeError("模型文件校验失败")
    return model_dir


def _default_embed() -> EmbedFn:
    """默认 embedding: fastembed 本地 ONNX 模型(首次运行下载, 之后走缓存)。

    镜像环境变量已在模块顶部设置; 模型缓存固定在用户目录, 避免依赖 Temp。
    """
    from fastembed import TextEmbedding  # 延迟导入: 不让 import 本模块就加载模型

    model_dir = _ensure_model()
    model = TextEmbedding(
        "BAAI/bge-small-zh-v1.5", specific_model_path=str(model_dir)
    )

    def embed(texts: list[str]) -> list[list[float]]:
        # tolist(): numpy float32 -> Python float(chroma 不接受 np 标量)
        return [v.tolist() for v in model.embed(texts)]

    return embed


def subdivide(chunk: Chunk, window: int = _WINDOW) -> list[tuple[int, str]]:
    """章节 chunk -> [(页码, 子块文本)] 子块列表。

    先按换页符 \\f 拆页(保证子块不跨页), 再按窗口长度切。
    返回页码用于 Citation 溯源。
    """
    pieces: list[tuple[int, str]] = []
    for offset, page_text in enumerate(chunk.text.split("\f")):
        page = chunk.page + offset
        text = page_text.strip()
        if not text:
            continue
        for i in range(0, len(text), window):
            pieces.append((page, text[i : i + window]))
    return pieces


class ReportVectorStore:
    def __init__(
        self,
        client: chromadb.ClientAPI | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.client = client or chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection("report_chunks")
        self.embed_fn = embed_fn or _default_embed()

    def add_chunks(self, code: str, period: str, chunks: list[Chunk]) -> int:
        """子块入库, 返回入库数量。重复索引同一报告会先清旧数据。"""
        # 幂等: 同一(code, period)重复入库时清掉旧子块
        self.collection.delete(where={"$and": [{"code": code}, {"period": period}]})

        rows: list[tuple[int, str]] = []
        for c in chunks:
            rows.extend(subdivide(c))
        if not rows:
            return 0

        vectors = self.embed_fn([text for _page, text in rows])
        ids = [f"{code}-{period}-{i}" for i in range(len(rows))]
        # 子块继承其来源 chunk 的章节名, 页码来自 subdivide 的页边界追踪
        metadatas: list[dict] = []
        idx = 0
        for c in chunks:
            for _page, _text in subdivide(c):
                metadatas.append(
                    {
                        "code": code,
                        "period": period,
                        "page": _page,
                        "section": c.section,
                        "chunk_index": idx,
                    }
                )
                idx += 1

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[text for _page, text in rows],
            metadatas=metadatas,
        )
        return len(rows)

    def retrieve(
        self,
        question: str,
        code: str,
        period: str | None = None,
        top_k: int = 5,
    ) -> list[Citation]:
        """向量检索, 返回带页码溯源的引用(不做生成)。"""
        docs, metas = self._search(question, code, period, top_k)
        return [
            Citation(page=m["page"], section=m["section"], snippet=d[:100].replace("\n", " "))
            for d, m in zip(docs, metas)
        ]

    def _search(
        self,
        question: str,
        code: str,
        period: str | None,
        top_k: int,
    ) -> tuple[list[str], list[dict]]:
        """底层检索: 返回 (子块文本列表, 元数据列表)。"""
        where: dict = {"code": code}
        if period:
            where = {"$and": [{"code": code}, {"period": period}]}
        query_vec = self.embed_fn([_QUERY_PREFIX + question])[0]
        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
        )
        return res["documents"][0], res["metadatas"][0]

    def query_with_citations(
        self,
        question: str,
        code: str,
        period: str | None = None,
        top_k: int = 5,
        chat_fn: ChatFn | None = None,
    ) -> tuple[str, list[Citation]]:
        """检索 + 生成答案, 返回 (答案, 引用列表)。"""
        chat_fn = chat_fn or _default_chat
        docs, metas = self._search(question, code, period, top_k)
        if not docs:
            return "未在报告中找到相关内容。", []

        citations = [
            Citation(page=m["page"], section=m["section"], snippet=d[:100].replace("\n", " "))
            for d, m in zip(docs, metas)
        ]
        context = "\n\n".join(
            f"【第{m['page']}页 · {m['section']}】\n{d}" for d, m in zip(docs, metas)
        )
        system = (
            "你是A股财报分析助手。只依据给定的报告原文片段回答问题, "
            "并在引用事实后标注来源页码, 格式如 (第X页)。"
            "如果片段中没有足够信息, 直接说明报告中未找到, 禁止编造。"
        )
        user = f"报告原文片段:\n\n{context}\n\n问题: {question}"
        answer = chat_fn(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return answer, citations
