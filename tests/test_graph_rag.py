# -*- coding: utf-8 -*-
"""graph + RAG 集成测试: 注入假 fetcher / chat / store, 全离线。

覆盖:
- index 节点入库调用(与 debate 并行, 但不依赖其结果)
- 带 question 路由到 rag_qa, 答案+引用渲染进报告
- 无 question 时跳过 rag_qa 但仍入库
- 入库失败 / 问答失败 均不阻断主流程
- enable_rag=False 时零开销(不装配 RAG 节点)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.agents.graph import build_graph
from finance_agent.rag.store import Citation

PDF = Path(__file__).resolve().parents[1] / "data_cache/reports/600519/2025-年报.pdf"

_FAKE_REPLIES = {
    "合并资产负债表": '{"total_assets": 1000, "total_liabilities": 600, "total_equity": 400, "monetary_funds": 100}',
    "合并利润表": '{"revenue": 500, "operating_cost": 300, "net_profit": 120}',
    "合并现金流量表": '{"operating_cash_flow": 150}',
}


def _fake_fetcher():
    fetcher = MagicMock()
    fetcher.download.return_value = PDF
    return fetcher


def _fake_chat(messages: list[dict]) -> str:
    user_text = messages[-1]["content"]
    for name, reply in _FAKE_REPLIES.items():
        if name in user_text:
            return reply
    # 辩论/RAG 轮: 返回一段固定叙述
    return "测试回答"


class FakeStore:
    """记录调用的假向量库, query_with_citations 返回固定引用。"""

    def __init__(self, fail_add=False, fail_query=False):
        self.added: list[tuple[str, str, int]] = []  # (code, period, chunk数)
        self.queries: list[tuple[str, str]] = []
        self.fail_add = fail_add
        self.fail_query = fail_query

    def add_chunks(self, code, period, chunks) -> int:
        if self.fail_add:
            raise RuntimeError("向量库写入失败")
        self.added.append((code, period, len(chunks)))
        return len(chunks) * 3  # 假设每个 chunk 拆 3 个子块

    def query_with_citations(self, question, code, period=None, top_k=5, chat_fn=None):
        if self.fail_query:
            raise RuntimeError("检索失败")
        self.queries.append((question, code))
        citations = [Citation(page=77, section="存货", snippet="自制半成品 275 亿...")]
        return "存货由自制半成品构成 (第77页)", citations


def _run(question=None, store=None, enable_rag=True, **store_kw):
    if store is None:
        store = FakeStore(**store_kw)
    graph = build_graph(
        fetcher=_fake_fetcher(), chat_fn=_fake_chat,
        enable_rag=enable_rag, store=store,
    )
    state: dict = {"code": "600519", "period": "2025-年报"}
    if question:
        state["question"] = question
    return graph.invoke(state), store


def test_index_and_qa_with_question():
    """带 question: 入库 + 问答, 答案和引用都渲染进报告。"""
    state, store = _run(question="存货主要由什么构成?")
    assert state["indexed"] is True
    assert state["indexed_count"] > 0
    # 入库参数正确
    assert store.added and store.added[0][0] == "600519"
    assert store.added[0][1] == "2025-年报"
    # 问答发生且路由参数正确
    assert store.queries == [("存货主要由什么构成?", "600519")]
    qa = state["qa"]
    assert qa["question"] == "存货主要由什么构成?"
    assert "第77页" in qa["answer"]
    assert qa["citations"][0]["page"] == 77
    # 报告渲染
    assert "## 附注问答" in state["report_md"]
    assert "存货由自制半成品构成" in state["report_md"]
    assert "第77页" in state["report_md"]
    # 主流程不受影响
    assert state["ratios"]


def test_index_without_question():
    """无 question: 仍入库(供后续问答用), 但跳过 rag_qa, 报告无问答章节。"""
    state, store = _run()
    assert state["indexed"] is True
    assert store.added
    assert not store.queries
    assert "qa" not in state
    assert "## 附注问答" not in state["report_md"]


# ---------------------------------------------------------------- 失败降级


def test_index_failure_silent():
    """入库失败: 静默跳过, 主流程不受影响。"""
    state, store = _run(fail_add=True)
    assert "indexed" not in state
    assert "error" not in state or state.get("error") is None
    assert state["ratios"]  # 分析照常


def test_qa_failure_degrades():
    """问答失败: 错误写进 qa.answer, 主流程不受影响。"""
    state, _ = _run(question="任何问题", fail_query=True)
    assert "暂不可用" in state["qa"]["answer"]
    assert state["qa"]["citations"] == []
    assert state["ratios"]
    assert "## 附注问答" in state["report_md"]


# ---------------------------------------------------------------- 开关


def test_rag_disabled_zero_overhead():
    """enable_rag=False: 不装配 RAG 节点, store 完全不被触碰。"""
    state, store = _run(enable_rag=False, question="还会问吗?")
    assert "indexed" not in state
    assert not store.added and not store.queries
    assert "qa" not in state
    # 无 question 路径: report 仍正常产出
    assert "财报分析" in state["report_md"]


def test_rag_disabled_route_still_works():
    """enable_rag=False 但带 question: 图正常走完(问答被忽略, 不报错)。"""
    state, _ = _run(enable_rag=False, question="存货构成?")
    assert state["report_md"]
    assert "qa" not in state
