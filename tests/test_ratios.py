"""比率计算单元测试——评测基本盘, 第 3 周就要跑通 CI。"""

from finance_agent.analysis.ratios import compute_ratios, detect_anomalies
from finance_agent.parsing.extractor import (
    BalanceSheet,
    CashFlowStatement,
    ExtractedReport,
    IncomeStatement,
)


def make_report(
    revenue=100.0, net_profit=10.0, assets=200.0, liabilities=80.0, receivable=20.0
) -> ExtractedReport:
    return ExtractedReport(
        balance_sheet=BalanceSheet(
            total_assets=assets,
            total_liabilities=liabilities,
            total_equity=assets - liabilities,
            monetary_funds=30.0,
            accounts_receivable=receivable,
            inventory=15.0,
        ),
        income_statement=IncomeStatement(
            revenue=revenue,
            operating_cost=60.0,
            net_profit=net_profit,
            net_profit_deducted=9.0,
            rd_expense=5.0,
        ),
        cash_flow_statement=CashFlowStatement(
            operating_cash_flow=8.0, investing_cash_flow=-12.0, financing_cash_flow=2.0
        ),
    )


def test_compute_ratios_count():
    ratios = compute_ratios(make_report())
    names = [r.name for r in ratios]
    assert "净资产收益率 ROE" in names
    assert "资产负债率" in names
    assert "毛利率" in names
    assert "经营现金流/净利润" in names


def test_roe_value():
    ratios = {r.name: r for r in compute_ratios(make_report())}
    assert abs(ratios["净资产收益率 ROE"].value - 10.0 / 120.0) < 1e-9
    assert abs(ratios["资产负债率"].value - 0.4) < 1e-9


def test_anomaly_detection_triggers():
    anomalies = detect_anomalies(make_report(), make_report(receivable=5.0))
    assert any(a.metric == "应收账款" for a in anomalies)


def test_no_false_anomaly():
    assert detect_anomalies(make_report(), make_report()) == []
