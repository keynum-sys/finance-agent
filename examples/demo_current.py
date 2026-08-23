# -*- coding: utf-8 -*-
"""当前能力演示: 用模拟财报数据跑通"抽取模型 -> 比率分析 -> 异常检测 -> 会计校验"。

用法:
    python examples/demo_current.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.analysis.ratios import compute_ratios, detect_anomalies
from finance_agent.parsing.extractor import (
    BalanceSheet,
    CashFlowStatement,
    ExtractedReport,
    IncomeStatement,
)


def build_report(year: int) -> ExtractedReport:
    """构造某虚构公司 X 年度财报(单位: 元)。两年数据刻意制造了几处异常。"""
    if year == 2024:
        return ExtractedReport(
            source_pages={"balance_sheet": 8, "income_statement": 11, "cash_flow_statement": 14},
            balance_sheet=BalanceSheet(
                total_assets=10_000_000_000,   # 100亿
                total_liabilities=4_500_000_000,
                total_equity=5_500_000_000,
                monetary_funds=2_000_000_000,
                accounts_receivable=800_000_000,
                inventory=1_200_000_000,
            ),
            income_statement=IncomeStatement(
                revenue=6_000_000_000,
                operating_cost=3_600_000_000,
                net_profit=900_000_000,
                net_profit_deducted=850_000_000,
                rd_expense=300_000_000,
            ),
            cash_flow_statement=CashFlowStatement(
                operating_cash_flow=1_100_000_000,
                investing_cash_flow=-500_000_000,
                financing_cash_flow=-200_000_000,
            ),
        )
    # 2025 年: 营收微增, 但应收账款和存货激增, 经营现金流恶化 -> 应触发预警
    return ExtractedReport(
        source_pages={"balance_sheet": 8, "income_statement": 11, "cash_flow_statement": 14},
        balance_sheet=BalanceSheet(
            total_assets=10_800_000_000,
            total_liabilities=4_900_000_000,
            total_equity=5_900_000_000,
            monetary_funds=1_200_000_000,
            accounts_receivable=2_100_000_000,   # +162% -> 异常
            inventory=2_000_000_000,             # +67%  -> 异常
        ),
        income_statement=IncomeStatement(
            revenue=6_600_000_000,
            operating_cost=3_900_000_000,
            net_profit=1_050_000_000,
            net_profit_deducted=980_000_000,
            rd_expense=350_000_000,
        ),
        cash_flow_statement=CashFlowStatement(
            operating_cash_flow=300_000_000,     # 大幅下滑 -> 现金流/净利润 < 1
            investing_cash_flow=-400_000_000,
            financing_cash_flow=100_000_000,
        ),
    )


def main() -> None:
    report_2024 = build_report(2024)
    report_2025 = build_report(2025)

    print("=" * 62)
    print("FinSight 当前能力演示 (纯规则引擎, 无 LLM 依赖)")
    print("=" * 62)

    print("\n[1] 会计恒等式校验 (资产 = 负债 + 权益, 允许 0.5% 误差)")
    for r, y in [(report_2024, 2024), (report_2025, 2025)]:
        ok = r.check_accounting_identity()
        print(f"    {y} 年报: {'通过' if ok else '未通过 -> 标记 low_confidence 并重试抽取'}")

    print("\n[2] 核心财务比率 (2025 年报, 单位换算为亿元方便阅读)")
    for ratio in compute_ratios(report_2025):
        page = f"  (见第 {report_2025.source_pages.get('balance_sheet', '?')} 页)" if ratio.source_page else ""
        print(f"    {ratio.name:　<10s} {ratio.value:>8.2%}  {ratio.interpretation}{page}")

    print("\n[3] 同比异常检测 (2025 vs 2024, 阈值 ±30%)")
    anomalies = detect_anomalies(report_2025, report_2024)
    if anomalies:
        for a in anomalies:
            direction = "↑" if a.change_rate > 0 else "↓"
            print(f"    [预警] {a.metric}: {a.previous/1e8:.1f}亿 -> {a.current/1e8:.1f}亿"
                  f" ({a.change_rate:+.1%}) {direction}")
            print(f"           {a.note}")
    else:
        print("    未发现异常")

    print("\n[4] 溯源信息 (每条结论可回溯到财报原文页码)")
    for field, page in report_2025.source_pages.items():
        print(f"    {field} -> 财报第 {page} 页")

    print("\n结论: 净利润增长 16.7%, 但应收账款 +162%、存货 +67%、经营现金流")
    print("      从 11 亿降至 3 亿 —— 典型'纸面利润', 规则引擎已自动标记。")


if __name__ == "__main__":
    main()
