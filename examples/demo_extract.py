# -*- coding: utf-8 -*-
"""端到端验证: PDF -> 定位三大报表 -> LLM 抽取 -> 财务比率分析。

用法:
    python examples/demo_extract.py 600519
    python examples/demo_extract.py 000001
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.analysis.ratios import compute_ratios
from finance_agent.parsing.extractor import extract_report

REPORTS = {
    "600519": ("贵州茅台 2025 年报", "data_cache/reports/600519/2025-年报.pdf"),
    "000001": ("平安银行 2026 一季报", "data_cache/reports/000001/2026-一季报.pdf"),
}


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1e8:
        return f"{v / 1e8:,.2f} 亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:,.2f} 万"
    return f"{v:,.2f}"


def main() -> None:
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    if code not in REPORTS:
        print(f"未知代码 {code}, 可选: {list(REPORTS)}")
        sys.exit(1)
    name, pdf = REPORTS[code]
    print("=" * 60)
    print(f"端到端抽取验证: {name}")
    print("=" * 60)

    report = extract_report(pdf)

    print(f"\n置信度: {report.confidence}")
    print(f"恒等式校验(资产=负债+权益): {'通过' if report.check_accounting_identity() else '未通过'}")
    print(f"溯源页码: {report.source_pages}")

    print("\n--- 合并资产负债表 ---")
    bs = report.balance_sheet
    if bs:
        for k, v in bs.model_dump().items():
            print(f"  {k:24s} {fmt(v)}")
    else:
        print("  [抽取失败]")

    print("\n--- 合并利润表 ---")
    ic = report.income_statement
    if ic:
        for k, v in ic.model_dump().items():
            print(f"  {k:24s} {fmt(v)}")
    else:
        print("  [抽取失败]")

    print("\n--- 合并现金流量表 ---")
    cf = report.cash_flow_statement
    if cf:
        for k, v in cf.model_dump().items():
            print(f"  {k:24s} {fmt(v)}")
    else:
        print("  [抽取失败]")

    print("\n--- 财务比率(规则计算) ---")
    for r in compute_ratios(report):
        print(f"  {r.name:16s} {r.value:8.2%}   {r.interpretation}")

    print("\n完成。")


if __name__ == "__main__":
    main()
