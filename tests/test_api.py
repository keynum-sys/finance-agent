"""API 层离线测试: TestClient + 全部依赖注入, 不联网。

FakeStore 复用 test_graph_rag 的思路(有 add_chunks / query_with_citations)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from finance_agent.api.app import create_app


PDF = Path(__file__).resolve().parents[1] / "data_cache" / "reports" / "600519" / "2025-年报.pdf"

_FAKE_REPLIES = {
    "合并资产负债表": '{"total_assets": 1000, "total_liabilities": 600, "total_equity": 400, "monetary_funds": 100}',
    "合并利润表": '{"revenue": 500, "operating_cost": 300, "net_profit": 120}',
    "合并现金流量表": '{"operating_cash_flow": 150}',
}


# ---------------------------------------------------------------- 测试替身

class FakeFetcher:
    """下载: 直接给缓存的茅台 PDF(不联网, 只读本地文件)。"""

    def download(self, code: str, period: str) -> str:
        return str(PDF)


class FakeFailFetcher:
    def download(self, code: str, period: str) -> str:
        raise RuntimeError("下载失败")


def _fake_chat(messages, **kwargs):  # noqa: ANN001, ANN202
    user_text = messages[-1]["content"]
    for name, reply in _FAKE_REPLIES.items():
        if name in user_text:
            return reply
    return "好的, 这是测试回答。"


class _Cite:
    def __init__(self, page: int, section: str, snippet: str):
        self.page = page
        self.section = section
        self.snippet = snippet


class FakeStore:
    def __init__(self):
        self.added = []

    def add_chunks(self, code, period, chunks):  # noqa: ANN001, ANN202
        self.added.append((code, period, len(chunks)))
        return len(chunks)

    def query_with_citations(self, question, code, period=None, top_k=5, chat_fn=None):  # noqa: ANN001, ANN202
        return (
            "应收账款按组合计提坏账准备。",
            [_Cite(85, "财务报表附注", "按账龄组合计提...")],
        )


def _client(fetcher=None, store=None):
    app = create_app(fetcher=fetcher or FakeFetcher(), chat_fn=_fake_chat, store=store)
    return TestClient(app)


# ---------------------------------------------------------------- /health

def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------- /analyze

def test_analyze_full_pipeline():
    """完整流水线: 返回报告 + 比率 + 辩论(全 fake, 离线)。"""
    r = _client().post("/analyze", json={"code": "600519", "period": "2025-年报"})
    assert r.status_code == 200
    d = r.json()
    assert d["error"] is None
    assert "茅台" in d["report_md"] or "600519" in d["report_md"]
    assert len(d["ratios"]) > 0
    assert d["debate"] is not None
    assert d["qa"] is None  # 没传 question -> 无附注问答


def test_analyze_with_question_enables_rag():
    store = FakeStore()
    r = _client(store=store).post(
        "/analyze",
        json={
            "code": "600519",
            "period": "2025-年报",
            "question": "坏账准备怎么计提?",
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d["qa"] is not None
    assert d["qa"]["answer"] == "应收账款按组合计提坏账准备。"
    assert d["qa"]["citations"][0]["page"] == 85
    assert store.added  # index 节点确实入库了


def test_analyze_download_failure_degrades():
    """下载失败不抛 500: 流水线降级出错误报告, error 字段带原因。"""
    r = _client(fetcher=FakeFailFetcher()).post(
        "/analyze", json={"code": "999999", "period": "2025-年报"}
    )
    assert r.status_code == 200
    d = r.json()
    assert d["error"]
    assert d["report_md"]  # 仍返回降级版报告


def test_analyze_validation_error():
    """缺必填字段 -> 422, 不是 500。"""
    r = _client().post("/analyze", json={"code": "600519"})
    assert r.status_code == 422


# ---------------------------------------------------------------- /ask

def test_ask_pure_rag():
    store = FakeStore()
    r = _client(store=store).post(
        "/ask",
        json={"code": "600519", "period": "2025-年报", "question": "坏账政策?"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["answer"] == "应收账款按组合计提坏账准备。"
    assert d["citations"] == [
        {"page": 85, "section": "财务报表附注", "snippet": "按账龄组合计提..."}
    ]


def test_ask_validation_error():
    r = _client().post("/ask", json={"code": "600519", "question": "x"})
    assert r.status_code == 422
