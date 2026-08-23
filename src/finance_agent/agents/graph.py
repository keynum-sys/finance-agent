"""LangGraph 流水线编排: parse -> extract -> validate -> analyze -> debate -> report。

第 4 周搭 MVP(前四个节点), 第 8 周加 debate 节点。
选型理由: 财报分析是确定性流程, 图编排比 ReAct 自由循环更可控、可回溯。

TODO(第4周):
- 定义 ReportState(TypedDict): chunks / extracted / ratios / anomalies / report_md
- 每个节点一个函数, StateGraph 串起来
- validate 节点: 恒等式校验 + AKShare 交叉核对, 失败走重试边
- debate 节点(第8周): 财务分析师 vs 质疑者(审计视角) -> 裁判综合
"""

from __future__ import annotations

from typing import TypedDict


class ReportState(TypedDict, total=False):
    code: str                  # 股票代码
    pdf_path: str
    chunks: list               # list[Chunk]
    extracted: dict            # ExtractedReport.model_dump()
    ratios: list               # list[RatioResult]
    anomalies: list            # list[Anomaly]
    retry_count: int
    report_md: str             # 最终 Markdown 报告


def build_graph():
    """构建并编译 LangGraph 流水线。TODO(第4周) 实现。"""
    raise NotImplementedError
