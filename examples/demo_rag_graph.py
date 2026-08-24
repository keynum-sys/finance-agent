# -*- coding: utf-8 -*-
"""完整流水线 + RAG 附注问答真实验证: 一张图跑完分析全流程并回答附注问题。

复用缓存的茅台 2025 年报, 走真实 LangGraph 图:
download -> extract -> analyze -> {debate ∥ index} -> rag_qa -> report

用法: python examples/demo_rag_graph.py [问题]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.agents.graph import run_pipeline

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "应收账款的坏账准备是怎么计提的?"


def main() -> None:
    print("=" * 60)
    print(f"问题: {QUESTION}")
    print("=" * 60)

    state = run_pipeline("600519", "2025-年报", question=QUESTION)

    print(f"\n入库子块数: {state.get('indexed_count', 'N/A')}")
    qa = state.get("qa")
    if qa:
        print(f"\n--- 答案 ---\n{qa['answer']}")
        print("\n--- 引用来源 ---")
        for c in qa["citations"]:
            print(f"  第{c['page']}页({c['section']}): {c['snippet'][:60]}...")

    # 报告章节结构
    md = state["report_md"]
    print("\n--- 报告章节 ---")
    for line in md.splitlines():
        if line.startswith("#"):
            print(" ", line)

    out = Path(__file__).resolve().parents[1] / "output" / "moutai_rag_graph_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n完整报告已写入: {out}")


if __name__ == "__main__":
    main()
