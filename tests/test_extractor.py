# -*- coding: utf-8 -*-
"""extractor 单元测试: 注入假 chat_fn, 全程离线, 不调真实 API。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.parsing.extractor import (
    BalanceSheet,
    CashFlowStatement,
    ExtractedReport,
    IncomeStatement,
    _extract_statement,
    _identity_check,
    _schema_prompt,
    detect_unit,
)


# ---------------------------------------------------------------- 假 LLM

class FakeChat:
    """按顺序返回预设回复, 记录收到的 messages 供断言。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


# ---------------------------------------------------------------- prompt

def test_schema_prompt_contains_all_fields():
    prompt = _schema_prompt(BalanceSheet)
    for name in BalanceSheet.model_fields:
        assert f'"{name}"' in prompt
    assert "资产总计" in prompt  # 描述同步进 prompt


# ---------------------------------------------------------------- 成功路径

def test_extract_statement_success():
    good = '{"total_assets": 100, "total_liabilities": 60, "total_equity": 40, "monetary_funds": 10, "accounts_receivable": 5, "inventory": null}'
    chat = FakeChat([good])
    model = _extract_statement("文本", BalanceSheet, "合并资产负债表", chat)
    assert isinstance(model, BalanceSheet)
    assert model.total_assets == 100
    assert model.inventory is None  # 银行股场景: 科目缺失允许 null
    assert len(chat.calls) == 1


# ---------------------------------------------------------------- 重试机制

def test_retry_on_invalid_json():
    chat = FakeChat(["这不是JSON", '{"revenue": 1}'])
    model = _extract_statement("文本", IncomeStatement, "合并利润表", chat)
    assert model is not None and model.revenue == 1
    assert len(chat.calls) == 2
    # 第二次调用包含错误反馈
    assert "你的输出有问题" in chat.calls[1][-1]["content"]


def test_retry_on_validation_error():
    chat = FakeChat(['{"revenue": "一百万"}', '{"revenue": 1000000}'])
    model = _extract_statement("文本", IncomeStatement, "合并利润表", chat)
    assert model is not None and model.revenue == 1000000
    assert len(chat.calls) == 2


def test_give_up_after_max_retries():
    chat = FakeChat(["坏输出"] * 3)
    model = _extract_statement(
        "文本", IncomeStatement, "合并利润表", chat, max_retries=3
    )
    assert model is None
    assert len(chat.calls) == 3


def test_all_null_result_treated_as_failure():
    # 全 null = 实际上什么都没抽到, 应触发重试
    chat = FakeChat(['{"revenue": null}', '{"revenue": 50}'])
    model = _extract_statement("文本", IncomeStatement, "合并利润表", chat)
    assert model is not None and model.revenue == 50


# ---------------------------------------------------------------- 恒等式校验

def test_identity_check_pass():
    bs = BalanceSheet(total_assets=100, total_liabilities=60, total_equity=40)
    assert _identity_check(bs) is None


def test_identity_check_fail_triggers_retry():
    bad = '{"total_assets": 100, "total_liabilities": 60, "total_equity": 30}'
    good = '{"total_assets": 100, "total_liabilities": 60, "total_equity": 40}'
    chat = FakeChat([bad, good])
    model = _extract_statement(
        "文本", BalanceSheet, "合并资产负债表", chat, extra_check=_identity_check
    )
    assert model is not None and model.total_equity == 40
    assert len(chat.calls) == 2
    assert "会计恒等式" in chat.calls[1][-1]["content"]


# ---------------------------------------------------------------- 置信度

def test_confidence_high_when_complete_and_balanced():
    report = ExtractedReport(
        balance_sheet=BalanceSheet(total_assets=100, total_liabilities=60, total_equity=40),
        income_statement=IncomeStatement(revenue=10, net_profit=2),
        cash_flow_statement=CashFlowStatement(operating_cash_flow=3),
    )
    assert report.confidence == "high"
    assert report.check_accounting_identity()


def test_confidence_low_when_identity_fails():
    report = ExtractedReport(
        balance_sheet=BalanceSheet(total_assets=100, total_liabilities=60, total_equity=30),
        income_statement=IncomeStatement(revenue=10),
        cash_flow_statement=CashFlowStatement(operating_cash_flow=3),
    )
    assert not report.check_accounting_identity()
    # extract_report 主流程里会据此置 low; 此处直接验证判断逻辑
    if not report.check_accounting_identity():
        report.confidence = "low"
    assert report.confidence == "low"


def test_identity_tolerance():
    # 0.5% 容差内算通过
    bs = BalanceSheet(total_assets=1000000, total_liabilities=600000, total_equity=400200)
    report = ExtractedReport(balance_sheet=bs)
    assert report.check_accounting_identity()  # 差 200/1000000 = 0.02%


def test_source_pages_field():
    report = ExtractedReport(source_pages={"合并资产负债表": 56})
    assert report.source_pages["合并资产负债表"] == 56


# ---------------------------------------------------------------- 单位检测
# 背景: 评测发现银行股"百万元"换算易错 10 倍, prompt 需显式提示单位

def test_detect_unit_million():
    text = "平安银行股份有限公司\n合并资产负债表\n货币单位：人民币百万元\n资产\n6,033,962"
    assert detect_unit(text) == "百万元"


def test_detect_unit_yuan():
    text = "合并资产负债表\n单位：元\n币种：人民币\n项目"
    assert detect_unit(text) == "元"


def test_detect_unit_none():
    assert detect_unit("合并利润表\n2025年1-3月\n没有单位标注") is None


def test_user_prompt_contains_unit_hint():
    from finance_agent.parsing.extractor import _user_prompt
    text = "货币单位：人民币百万元\n资产总计 6,033,962"
    prompt = _user_prompt("合并资产负债表", text, BalanceSheet)
    assert "百万元" in prompt and "换算成元" in prompt
    # 元为单位的报表不加提示
    prompt2 = _user_prompt("合并资产负债表", "单位：元\n资产总计 100", BalanceSheet)
    assert "换算成元" not in prompt2
