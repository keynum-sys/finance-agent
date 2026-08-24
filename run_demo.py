# -*- coding: utf-8 -*-
"""FinSight 财报分析 Agent — 交互式一键启动器。

双击项目根目录下的「运行demo.bat」即可启动本脚本。
按提示输入股票代码与报告期，即可跑完整分析流水线并回答附注问题。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from finance_agent.agents.graph import run_pipeline

# 预设题库: 覆盖分红/现金流/负债/营收/研发等多维度, 随机抽题时选用
QUESTION_BANK = [
    "公司的现金分红政策是什么？本期分红预案如何？",
    "应收账款的坏账准备是怎么计提的？账龄分布如何？",
    "本期经营活动现金流净额是多少？与净利润是否匹配？",
    "公司的资产负债率和有息负债情况如何？",
    "营业收入同比/环比增长情况如何？主要驱动因素是什么？",
    "研发投入占营业收入的比例是多少？同比变化如何？",
    "主营业务毛利率是多少？相比上年有何变化？",
    "是否存在大额商誉？商誉减值风险如何？",
    "前五大客户/供应商的集中度如何？",
    "关联交易的主要内容和规模是什么？",
    "员工总数和人均薪酬变化情况如何？",
    "货币资金是否存在受限情形？受限金额多少？",
]


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
    raw = input("可选: 输入附注问题 (直接回车或输入'随机' = 随机抽一题): ").strip()
    if not raw or raw in ("随机", "随机提问", "random", "r", "R"):
        question = random.choice(QUESTION_BANK)
        print(f"[随机抽题] {question}")
    else:
        question = raw

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
