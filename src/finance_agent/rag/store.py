"""Chroma 向量库: 章节级入库 + 带页码溯源的检索问答。

第 9 周实现。设计要点:
- chunk 入库时 metadata 携带 {code, period, page, section}
- 检索答案必须附带引用: [(page, section, 原文片段)]
- 报告生成节点同样从这里取引用 -> "每条结论可溯源"的落地
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from finance_agent.config import settings


@dataclass
class Citation:
    page: int
    section: str
    snippet: str


class ReportVectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection("report_chunks")

    def add_chunks(self, code: str, period: str, chunks) -> None:
        """章节 chunk 入库。TODO(第9周): embedding + metadata。"""
        raise NotImplementedError

    def query_with_citations(self, question: str, code: str, top_k: int = 5) -> tuple[str, list[Citation]]:
        """检索 + 生成答案, 返回 (答案, 引用列表)。TODO(第9周)。"""
        raise NotImplementedError
