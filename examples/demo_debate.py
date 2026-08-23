# -*- coding: utf-8 -*-
"""辩论节点真实验证: 复用缓存的茅台 2025 年报, 三轮辩论走真实 DeepSeek。

用法: python examples/demo_debate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.agents.debate import run_debate, default_chat
from finance_agent.parsing.extractor import extract_report

PDF = Path(__file__).resolve().parents[1] / "data_cache/reports/600519/2025-年报.pdf"


def main() -> None:
    print("=" * 60)
    print("1. LLM 抽取三大报表(带缓存特征: 直接解析已下载的 PDF)")
    report = extract_report(str(PDF))
    print(f"   置信度: {report.confidence}")

    # 比率
    from dataclasses import asdict
    from finance_agent.analysis.ratios import compute_ratios
    ratios = [asdict(r) for r in compute_ratios(report)]
    print(f"   财务比率: {len(ratios)} 项")
    for r in ratios:
        print(f"     - {r['name']}: {r['value']:.2%} ({r['interpretation']})")

    print("=" * 60)
    print("2. 三轮辩论(多头 -> 空头 -> 裁判), 真实 DeepSeek 调用中...")
    result = run_debate(report.model_dump(), ratios, default_chat)

    print("\n----- 多头论据 -----")
    print(result.bull_argument)
    print("\n----- 空头论据 -----")
    print(result.bear_argument)
    print(f"\n----- 裁决({result.stance}) -----")
    print(result.verdict)
    print("=" * 60)
    print("辩论完成")


if __name__ == "__main__":
    main()
