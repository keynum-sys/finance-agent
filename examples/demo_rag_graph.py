# -*- coding: utf-8 -*-
"""完整流水线 + RAG 附注问答真实验证: 一张图跑完分析全流程并回答附注问题。

走真实 LangGraph 图:
download -> extract -> analyze -> {debate ∥ index} -> rag_qa -> report

用法:
    python examples/demo_rag_graph.py <代码> <报告期> [问题]
    python examples/demo_rag_graph.py 600519 2025-年报 "分红政策是什么?"
    python examples/demo_rag_graph.py 300750 2025-年报           # 用默认问题

报告期格式: YYYY-年报 | YYYY-半年报 | YYYY-一季报 | YYYY-三季报
首次跑某家公司会联网下载 PDF 并缓存到 data_cache/reports/<代码>/<报告期>.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.agents.graph import run_pipeline

DEFAULT_QUESTION = "应收账款的坏账准备是怎么计提的?"

USAGE = (
    "用法: python examples/demo_rag_graph.py <代码> <报告期> [问题]\n"
    "示例: python examples/demo_rag_graph.py 600519 2025-年报 \"分红政策是什么?\"\n"
    "报告期: YYYY-年报 | YYYY-半年报 | YYYY-一季报 | YYYY-三季报"
)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(USAGE)
        sys.exit(1)

    code, period = args[0], args[1]
    question = args[2] if len(args) > 2 else DEFAULT_QUESTION

    print("=" * 60)
    print(f"公司: {code}  报告期: {period}")
    print(f"问题: {question}")
    print("=" * 60)

    state = run_pipeline(code, period, question=question)

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

    safe = f"{code}_{period}".replace("/", "_")
    out = Path(__file__).resolve().parents[1] / "output" / f"{safe}_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n完整报告已写入: {out}")


if __name__ == "__main__":
    main()
