# -*- coding: utf-8 -*-
"""FinSight 财报分析 Agent — 交互式一键启动器。

双击项目根目录下的「运行demo.bat」即可启动本脚本。
按提示输入股票代码与报告期，即可跑完整分析流水线并回答附注问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from finance_agent.agents.graph import run_pipeline

DEFAULT_QUESTION = "应收账款的坏账准备是怎么计提的?"


def ask(prompt: str, default: str) -> str:
    val = input(prompt).strip()
    return val or default


def main() -> None:
    print("=" * 60)
    print("FinSight 财报分析 Agent — 一键运行")
    print("=" * 60)
    print("示例公司代码: 600519(茅台) 000001(平安银行) 300750(宁德时代) 601318(中国平安)")
    print("报告期格式: YYYY-年报 | YYYY-半年报 | YYYY-一季报 | YYYY-三季报")
    print("-" * 60)

    code = ask("请输入股票代码 (如 600519，回车默认 600519): ", "600519")
    period = ask("请输入报告期 (如 2025-年报，回车默认 2025-年报): ", "2025-年报")
    question = input("可选: 输入附注问题 (直接回车用默认问题): ").strip()
    if not question:
        question = DEFAULT_QUESTION

    print(f"\n开始分析 {code} {period}，首次会联网下载 PDF，请稍候...\n")
    state = run_pipeline(code, period, question=question)

    print(f"\n入库子块数: {state.get('indexed_count', 'N/A')}")
    qa = state.get("qa")
    if qa:
        print(f"\n--- 答案 ---\n{qa['answer']}")
        print("\n--- 引用来源 ---")
        for c in qa["citations"]:
            print(f"  第{c['page']}页({c['section']}): {c['snippet'][:60]}...")

    md = state["report_md"]
    print("\n--- 报告章节 ---")
    for line in md.splitlines():
        if line.startswith("#"):
            print(" ", line)

    safe = f"{code}_{period}".replace("/", "_")
    out = Path(__file__).resolve().parent / "output" / f"{safe}_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n完整报告已写入: {out}")
    input("\n按回车键退出...")


if __name__ == "__main__":
    main()
