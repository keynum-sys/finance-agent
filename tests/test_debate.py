# -*- coding: utf-8 -*-
"""debate 单元测试: 注入假 chat_fn, 全离线验证三轮辩论与容错降级。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.agents.debate import DebateResult, run_debate, _data_digest
from finance_agent.agents.graph import build_graph, run_pipeline

# ---------------------------------------------------------------- 测试数据

_EXTRACTED = {
    "balance_sheet": {"total_assets": 1000, "total_liabilities": 600,
                      "total_equity": 400, "monetary_funds": 100},
    "income_statement": {"revenue": 500, "operating_cost": 300, "net_profit": 120},
    "cash_flow_statement": {"operating_cash_flow": 150},
    "confidence": "high",
}

_RATIOS = [
    {"name": "净资产收益率 ROE", "value": 0.3, "interpretation": "优秀"},
    {"name": "资产负债率", "value": 0.6, "interpretation": "偏高"},
]


def _fake_debate_chat(messages: list[dict]) -> str:
    """按 system prompt 分流: 多头/空头/裁判各返回预设回复。"""
    system = messages[0]["content"]
    if "首席投资官" in system:
        return '{"stance": "看多", "verdict": "盈利能力强劲, ROE 30%, 主要风险是杠杆。"}'
    if "空头" in system:
        return "- 资产负债率 60% 偏高\n- 经营现金流 150 元低于净利润 120 元? 存疑"
    return "- 净利润 120 元, ROE 30%\n- 货币资金 100 元充裕"


# ---------------------------------------------------------------- 数据摘要

def test_data_digest_contains_statements_and_ratios():
    digest = _data_digest(_EXTRACTED, _RATIOS)
    assert "balance_sheet" in digest
    assert "total_assets=1,000" in digest          # 千分位
    assert "比率 净资产收益率 ROE=30.00%" in digest
    assert "偏高" in digest                          # 比率解读


def test_data_digest_empty_input():
    assert "无可用财务数据" in _data_digest({}, [])


# ---------------------------------------------------------------- 三轮辩论

def test_run_debate_happy_path():
    result = run_debate(_EXTRACTED, _RATIOS, _fake_debate_chat)
    assert isinstance(result, DebateResult)
    assert result.stance == "看多"
    assert "ROE 30%" in result.verdict
    assert result.bull_argument.startswith("-")      # 条目列表
    assert result.bear_argument.startswith("-")


def test_bear_sees_bull_argument():
    """空头轮必须能看到多头论据(真正的对抗, 而非各说各话)。"""
    seen: list[str] = []

    def chat(messages: list[dict]) -> str:
        if "空头" in messages[0]["content"]:
            seen.append(messages[-1]["content"])
            return "- 驳斥"
        if "首席投资官" in messages[0]["content"]:
            return '{"stance": "中性", "verdict": "平局"}'
        return "- 多头论据A"

    run_debate(_EXTRACTED, _RATIOS, chat)
    assert "多头论据A" in seen[0]                    # 多头论据透传给了空头


def test_judge_sees_both_arguments():
    seen: list[str] = []

    def chat(messages: list[dict]) -> str:
        if "首席投资官" in messages[0]["content"]:
            seen.append(messages[-1]["content"])
            return '{"stance": "中性", "verdict": "平局"}'
        if "空头" in messages[0]["content"]:
            return "- 空头论据B"
        return "- 多头论据A"

    run_debate(_EXTRACTED, _RATIOS, chat)
    assert "多头论据A" in seen[0] and "空头论据B" in seen[0]


# ---------------------------------------------------------------- 裁决容错

def test_judge_invalid_json_then_retry():
    calls = {"judge": 0}

    def chat(messages: list[dict]) -> str:
        if "首席投资官" in messages[0]["content"]:
            calls["judge"] += 1
            if calls["judge"] == 1:
                return "我认为看多! (非JSON)"       # 第一次输出垃圾
            return '{"stance": "看空", "verdict": "风险偏大。"}'
        if "空头" in messages[0]["content"]:
            return "- 看空"
        return "- 看多"

    result = run_debate(_EXTRACTED, _RATIOS, chat, max_retries=3)
    assert calls["judge"] == 2                       # 第二次成功
    assert result.stance == "看空"


def test_judge_retry_gets_error_feedback():
    """重试时错误信息要回传给模型(与 extractor 相同的自我纠正机制)。"""
    feedbacks: list[str] = []

    def chat(messages: list[dict]) -> str:
        if "首席投资官" in messages[0]["content"]:
            if len(messages) > 2:                    # 重试轮: 里有 assistant 回复
                feedbacks.append(messages[-1]["content"])
                return '{"stance": "中性", "verdict": "ok"}'
            return "垃圾输出"
        return "- 论据"

    run_debate(_EXTRACTED, _RATIOS, chat, max_retries=2)
    assert any("输出有问题" in f for f in feedbacks)


def test_judge_always_invalid_degrades_gracefully():
    """裁决始终解析失败: 降级保留多空论据, stance 中性, verdict 记原文。"""

    def chat(messages: list[dict]) -> str:
        if "首席投资官" in messages[0]["content"]:
            return "永远不是JSON"
        if "空头" in messages[0]["content"]:
            return "- 看空"
        return "- 看多"

    result = run_debate(_EXTRACTED, _RATIOS, chat, max_retries=2)
    assert result.stance == "中性"
    assert "解析失败" in result.verdict
    assert "永远不是JSON" in result.verdict          # 原文保留
    assert result.bull_argument == "- 看多"           # 多空论据仍在


def test_judge_invalid_stance_rejected():
    """stance 只能取 看多/看空/中性, 其他值触发重试。"""

    def chat(messages: list[dict]) -> str:
        if "首席投资官" in messages[0]["content"]:
            if len(messages) == 2:
                return '{"stance": "强烈推荐买入", "verdict": "x"}'  # 非法枚举
            return '{"stance": "中性", "verdict": "y"}'
        return "- 论据"

    result = run_debate(_EXTRACTED, _RATIOS, chat, max_retries=2)
    assert result.stance == "中性"


# ---------------------------------------------------------------- 图集成

def _full_fake_chat(messages: list[dict]) -> str:
    """同时服务 extract 和 debate 两类节点的假 LLM。"""
    user_text = messages[-1]["content"]
    for name, reply in {
        "合并资产负债表": '{"total_assets": 1000, "total_liabilities": 600,'
                          ' "total_equity": 400}',
        "合并利润表": '{"revenue": 500, "net_profit": 120}',
        "合并现金流量表": '{"operating_cash_flow": 150}',
    }.items():
        if name in user_text:
            return reply
    return _fake_debate_chat(messages)               # 其余交给辩论分流


def _fake_fetcher():
    fetcher = MagicMock()
    fetcher.download.return_value = Path("data_cache/reports/600519/2025-年报.pdf")
    return fetcher


def test_graph_runs_debate_after_analyze():
    graph = build_graph(fetcher=_fake_fetcher(), chat_fn=_full_fake_chat)
    state = graph.invoke({"code": "600519", "period": "2025-年报"})
    assert state.get("ratios")                       # debate 前置条件成立
    assert state.get("debate")                       # 辩论结果入 state
    assert state["debate"]["stance"] == "看多"
    # 报告渲染了辩论章节
    md = state["report_md"]
    assert "多空辩论" in md
    assert "多头论据" in md and "空头论据" in md
    assert "裁决(看多)" in md
    assert "ROE 30%" in md


def test_debate_failure_does_not_break_pipeline():
    """辩论轮抛异常: 静默跳过, 报告照常生成(无辩论章节)。"""

    def chat(messages: list[dict]) -> str:
        if "首席投资官" in messages[0]["content"] or "财务分析师" in messages[0]["content"]:
            raise RuntimeError("LLM 挂了")
        return _full_fake_chat(messages)

    graph = build_graph(fetcher=_fake_fetcher(), chat_fn=chat)
    state = graph.invoke({"code": "600519", "period": "2025-年报"})
    assert "debate" not in state                     # 辩论被跳过
    assert state.get("report_md")                    # 主流程不受影响
    assert "多空辩论" not in state["report_md"]
    assert "净资产收益率 ROE" in state["report_md"]   # 比率仍在


def test_extraction_failure_skips_debate_too():
    """抽取全失败 -> 直接路由到 report, 不经过 debate。"""

    def chat(messages: list[dict]) -> str:
        return "垃圾输出"

    graph = build_graph(fetcher=_fake_fetcher(), chat_fn=chat)
    state = graph.invoke({"code": "600519", "period": "2025-年报"})
    assert "debate" not in state
    assert "ratios" not in state
    assert "财报分析" in state["report_md"]


def test_run_pipeline_with_debate():
    state = run_pipeline(
        "600519", "2025-年报", fetcher=_fake_fetcher(), chat_fn=_full_fake_chat
    )
    assert state["debate"]["stance"] in ("看多", "看空", "中性")
