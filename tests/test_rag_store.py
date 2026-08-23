"""rag/store 离线单元测试: 子分块 / 入库幂等 / 检索过滤 / 引用结构 / 答案生成。"""

import chromadb
import pytest

from finance_agent.parsing.pdf_parser import Chunk
from finance_agent.rag.store import (
    Citation,
    ReportVectorStore,
    _QUERY_PREFIX,
    subdivide,
)


def keyword_embed(texts: list[str]) -> list[list[float]]:
    """假 embedding: 按「主题词 -> 维度」映射, 同主题文本向量相近。

    维度 0 = 坏账相关, 维度 1 = 存货相关, 其余维度固定 0.01。
    查询前缀不影响结果。
    """

    def vec(t: str) -> list[float]:
        t = t.removeprefix(_QUERY_PREFIX)
        v = [0.01] * 8
        if "坏账" in t:
            v[0] = 1.0
        if "存货" in t:
            v[1] = 1.0
        return v

    return [vec(t) for t in texts]


@pytest.fixture()
def store():
    client = chromadb.EphemeralClient()
    return ReportVectorStore(client=client, embed_fn=keyword_embed)


def test_subdivide_respects_page_boundaries():
    chunk = Chunk(
        page=10,
        page_end=12,
        section="重要会计政策",
        text="第一页内容\n\f" + "第二页" + "长文本" * 400 + "\n\f第三页内容",
    )
    pieces = subdivide(chunk)
    pages = [p for p, _ in pieces]
    # 页边界: 10/11/12 各自独立, 不跨页
    assert set(pages) == {10, 11, 12}
    # 超长页被窗口切分, 每块 <= 500 字
    assert all(len(text) <= 500 for _p, text in pieces)
    assert sum(1 for p in pages if p == 11) > 1


def test_subdivide_empty_pages_skipped():
    chunk = Chunk(page=1, page_end=3, section="S", text="有内容\n\f\n\f又有内容")
    pieces = subdivide(chunk)
    assert [p for p, _t in pieces] == [1, 3]


def test_add_chunks_and_retrieve(store):
    chunks = [
        Chunk(page=80, page_end=80, section="金融工具减值", text="应收账款按预期信用损失模型计提坏账准备"),
        Chunk(page=90, page_end=90, section="存货", text="存货主要包括基酒和在制半成品"),
    ]
    n = store.add_chunks("600519", "2025-年报", chunks)
    assert n == 2

    hits = store.retrieve("坏账准备怎么计提", "600519")
    assert len(hits) == 2
    assert hits[0].section == "金融工具减值"
    assert hits[0].page == 80
    assert "坏账" in hits[0].snippet

    hits2 = store.retrieve("存货包括什么", "600519")
    assert hits2[0].section == "存货"


def test_retrieve_filters_by_code(store):
    chunks = [Chunk(page=1, page_end=1, section="S", text="坏账政策文本")]
    store.add_chunks("600519", "2025-年报", chunks)
    store.add_chunks("000001", "2026-一季报", chunks)
    # 只查 000001: 能命中; 查不存在的代码: 空
    assert len(store.retrieve("坏账", "000001")) == 1
    assert store.retrieve("坏账", "300750") == []


def test_add_chunks_idempotent(store):
    chunks = [Chunk(page=1, page_end=1, section="S", text="坏账政策文本")]
    store.add_chunks("600519", "2025-年报", chunks)
    store.add_chunks("600519", "2025-年报", chunks)  # 重复索引
    assert len(store.retrieve("坏账", "600519")) == 1  # 不重复


def test_query_with_citations_generation(store):
    chunks = [
        Chunk(page=80, page_end=80, section="金融工具减值", text="应收账款坏账准备按预期信用损失计提")
    ]
    store.add_chunks("600519", "2025-年报", chunks)

    captured: list[list[dict]] = []

    def fake_chat(messages: list[dict]) -> str:
        captured.append(messages)
        return "坏账准备按预期信用损失模型计提 (第80页)"

    answer, citations = store.query_with_citations(
        "坏账政策", "600519", chat_fn=fake_chat
    )
    assert answer == "坏账准备按预期信用损失模型计提 (第80页)"
    assert citations == [
        Citation(page=80, section="金融工具减值", snippet="应收账款坏账准备按预期信用损失计提")
    ]
    # 生成 prompt 里带页码标签和原文
    user_msg = captured[0][1]["content"]
    assert "【第80页 · 金融工具减值】" in user_msg
    assert "预期信用损失" in user_msg


def test_query_no_hits_returns_fallback(store):
    # 注意: chromadb.EphemeralClient 在同一进程内共享底层存储,
    # 这里用其他测试没用过的代码, 保证集合里没有它的数据
    answer, citations = store.query_with_citations("任意问题", "999999")
    assert "未在报告中找到" in answer
    assert citations == []
