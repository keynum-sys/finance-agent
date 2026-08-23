# -*- coding: utf-8 -*-
"""graph 单元测试: 注入假 fetcher / chat_fn, 全离线验证三条执行路径。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.agents.graph import build_graph, run_pipeline

# 假的 LLM 回复: 三张报表各一条
_FAKE_REPLIES = {
    "合并资产负债表": '{"total_assets": 1000, "total_liabilities": 600, "total_equity": 400, "monetary_funds": 100}',
    "合并利润表": '{"revenue": 500, "operating_cost": 300, "net_profit": 120}',
    "合并现金流量表": '{"operating_cash_flow": 150}',
}


def _fake_fetcher():
    fetcher = MagicMock()
    fetcher.download.return_value = Path("data_cache/reports/600519/2025-年报.pdf")
    return fetcher


def _fake_chat(messages: list[dict]) -> str:
    """根据 prompt 中出现的报表名返回对应 JSON。"""
    user_text = messages[-1]["content"]
    for name, reply in _FAKE_REPLIES.items():
        if name in user_text:
            return reply
    raise AssertionError(f"未知报表 prompt: {user_text[:50]}")


def _run(fetcher, chat_fn=None):
    graph = build_graph(fetcher=fetcher, chat_fn=chat_fn)
    return graph.invoke({"code": "600519", "period": "2025-年报"})


# ---------------------------------------------------------------- 成功路径


def test_happy_path():
    state = _run(_fake_fetcher(), _fake_chat)
    assert state.get("pdf_path")
    assert state["extracted"]["balance_sheet"]["total_assets"] == 1000
    # 恒等式通过 + 三表齐全 -> high
    assert state["extracted"]["confidence"] == "high"
    # 比率已算出
    names = [r["name"] for r in state["ratios"]]
    assert "净资产收益率 ROE" in names
    assert "资产负债率" in names
    # 报告包含溯源和比率表
    assert "财报分析" in state["report_md"]
    assert "第" in state["report_md"]          # 溯源页码(假数据无页码, 但模板含"第")
    assert "净资产收益率 ROE" in state["report_md"]
    assert "error" not in state or state.get("error") is None


def test_report_markdown_contains_statements():
    state = _run(_fake_fetcher(), _fake_chat)
    md = state["report_md"]
    assert "资产负债表要点" in md
    assert "利润表要点" in md
    assert "现金流量表要点" in md
    assert "1,000" in md                      # 千分位格式化


# ---------------------------------------------------------------- 失败分支


def test_download_failure_short_circuits():
    fetcher = MagicMock()
    fetcher.download.side_effect = RuntimeError("网络超时")
    state = _run(fetcher)
    # 不应进入抽取节点
    assert "extracted" not in state
    assert "下载失败" in state["error"]
    assert "执行失败" in state["report_md"]


def test_extraction_failure_goes_to_report():
    # chat_fn 永远返回垃圾 -> 三张报表全部抽取失败
    state = _run(_fake_fetcher(), lambda messages: "垃圾输出")
    assert state.get("error") is None           # extract 节点本身没崩
    extracted = state.get("extracted") or {}
    has_any = any(
        extracted.get(k) for k in
        ("balance_sheet", "income_statement", "cash_flow_statement")
    )
    assert not has_any                          # 三表全 None
    # 路由应跳过 analyze 直接出报告
    assert "ratios" not in state
    assert "财报分析" in state["report_md"]


def test_run_pipeline_convenience():
    fetcher = _fake_fetcher()
    state = run_pipeline("600519", "2025-年报", fetcher=fetcher, chat_fn=_fake_chat)
    assert "report_md" in state
