# -*- coding: utf-8 -*-
"""LangGraph 流水线端到端演示: download -> extract -> analyze -> report。

用法:
    python examples/demo_pipeline.py 600519 2025-年报
    python examples/demo_pipeline.py 000001 2026-一季报
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.agents.graph import run_pipeline


def main() -> None:
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    period = sys.argv[2] if len(sys.argv) > 2 else "2025-年报"

    print("=" * 60)
    print(f"LangGraph 流水线: {code} {period}")
    print("=" * 60)

    state = run_pipeline(code, period)

    if state.get("error"):
        print(f"\n[失败] {state['error']}")
        print(state.get("report_md", ""))
        sys.exit(1)

    print(state["report_md"])

    out = Path(f"output/{code}_{period}.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(state["report_md"], encoding="utf-8")
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
