"""LangGraph 流水线编排: download -> extract -> analyze -> report。

选型理由: 财报分析是确定性流程, 图编排比 ReAct 自由循环更可控、可回溯。

图结构(条件边处理失败分支):

    START -> download
    download --成功--> extract --抽取到数据--> analyze -> report -> END
        |                |
        +--失败----------+--全部失败--> report -> END

设计要点:
- 每个节点是一个纯函数: 读 state -> 返回部分更新(dict), 不抛异常
  (节点内异常转成 state["error"], 由条件边路由到 report 收尾)
- fetcher / chat_fn 依赖注入, 单元测试全离线
- report 节点目前是确定性模板(第 8 周换成 LLM 叙述 + 辩论节点)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from finance_agent.analysis.ratios import compute_ratios
from finance_agent.data.fetcher import ReportFetcher
from finance_agent.parsing.extractor import ChatFn, ExtractedReport, extract_report


class ReportState(TypedDict, total=False):
    """流水线共享状态。total=False: 所有键可选, 节点按需写入。"""

    # 输入
    code: str                   # 股票代码, 如 "600519"
    period: str                 # 报告期, 如 "2025-年报"
    # 中间产物
    pdf_path: str
    extracted: dict             # ExtractedReport.model_dump()
    ratios: list[dict]          # [RatioResult.model_dump()]
    # 结果
    report_md: str              # 最终 Markdown 报告
    error: str | None           # 任一节点失败的原因


# --------------------------------------------------------------------------
# 节点函数: 每个节点返回"部分状态更新"
# --------------------------------------------------------------------------


def _make_download_node(fetcher: ReportFetcher):
    def download(state: ReportState) -> dict:
        """下载财报 PDF(带缓存)。失败不抛异常, 写入 error 由路由决策。"""
        try:
            path = fetcher.download(state["code"], state["period"])
        except Exception as e:  # 网络/解析等任何失败都收敛到这里
            return {"error": f"下载失败: {e}"}
        return {"pdf_path": str(path)}

    return download


def _make_extract_node(chat_fn: ChatFn | None):
    def extract(state: ReportState) -> dict:
        """LLM 结构化抽取三大报表。"""
        if state.get("error"):
            return {}  # 上游已失败, 跳过
        try:
            report = extract_report(state["pdf_path"], chat_fn=chat_fn)
        except Exception as e:
            return {"error": f"抽取失败: {e}"}
        return {"extracted": report.model_dump()}

    return extract


def analyze(state: ReportState) -> dict:
    """规则引擎计算财务比率。extracted 缺失时由路由跳过, 不会进来。"""
    data = state.get("extracted") or {}
    report = ExtractedReport.model_validate(data)
    ratios = compute_ratios(report)
    return {"ratios": [asdict(r) for r in ratios]}


def report(state: ReportState) -> dict:
    """生成最终 Markdown 报告(确定性模板, 第 8 周换 LLM 叙述)。"""
    lines: list[str] = [f"# {state.get('code', '?')} {state.get('period', '')} 财报分析", ""]

    if state.get("error"):
        lines += [f"**执行失败**: {state['error']}", ""]
        return {"report_md": "\n".join(lines)}

    data = state.get("extracted") or {}
    lines.append(f"数据可信度: **{data.get('confidence', 'unknown')}**"
                 f"(会计恒等式{'通过' if data.get('balance_sheet') else '未校验'})")
    if data.get("source_pages"):
        pages = ", ".join(f"{k}(第{v}页)" for k, v in data["source_pages"].items())
        lines.append(f"溯源: {pages}")
    lines.append("")

    if data.get("balance_sheet"):
        lines += ["## 资产负债表要点", ""]
        for k, v in data["balance_sheet"].items():
            if v is not None:
                lines.append(f"- {k}: {v:,.0f} 元")
        lines.append("")
    if data.get("income_statement"):
        lines += ["## 利润表要点", ""]
        for k, v in data["income_statement"].items():
            if v is not None:
                lines.append(f"- {k}: {v:,.0f} 元")
        lines.append("")
    if data.get("cash_flow_statement"):
        lines += ["## 现金流量表要点", ""]
        for k, v in data["cash_flow_statement"].items():
            if v is not None:
                lines.append(f"- {k}: {v:,.0f} 元")
        lines.append("")

    ratios = state.get("ratios") or []
    if ratios:
        lines += ["## 财务比率", "", "| 指标 | 数值 | 解读 |", "|---|---|---|"]
        for r in ratios:
            lines.append(f"| {r['name']} | {r['value']:.2%} | {r['interpretation']} |")
        lines.append("")

    return {"report_md": "\n".join(lines)}


# --------------------------------------------------------------------------
# 路由函数(条件边的核心): 读 state 返回下一个节点名
# --------------------------------------------------------------------------


def _route_after_download(state: ReportState) -> str:
    """下载成功 -> 抽取; 失败 -> 直接出报告(带错误信息)。"""
    return "extract" if not state.get("error") else "report"


def _route_after_extract(state: ReportState) -> str:
    """抽到任一报表 -> 分析; 全军覆没 -> 出报告。"""
    data = state.get("extracted") or {}
    has_any = any(
        data.get(k) for k in ("balance_sheet", "income_statement", "cash_flow_statement")
    )
    return "analyze" if has_any and not state.get("error") else "report"


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------


def build_graph(
    fetcher: ReportFetcher | None = None,
    chat_fn: ChatFn | None = None,
) -> object:
    """构建并编译流水线图。

    fetcher / chat_fn 均可注入假实现, 测试全离线。
    返回编译后的图, 用 result = graph.invoke({"code": ..., "period": ...}) 运行。
    """
    fetcher = fetcher or ReportFetcher()

    g = StateGraph(ReportState)
    g.add_node("download", _make_download_node(fetcher))
    g.add_node("extract", _make_extract_node(chat_fn))
    g.add_node("analyze", analyze)
    g.add_node("report", report)

    g.add_edge(START, "download")
    g.add_conditional_edges(
        "download",
        _route_after_download,
        {"extract": "extract", "report": "report"},
    )
    g.add_conditional_edges(
        "extract",
        _route_after_extract,
        {"analyze": "analyze", "report": "report"},
    )
    g.add_edge("analyze", "report")
    g.add_edge("report", END)

    return g.compile()


def run_pipeline(code: str, period: str, **kwargs) -> ReportState:
    """便捷入口: 跑完整流水线, 返回最终 state(含 report_md)。"""
    graph = build_graph(**kwargs)
    return graph.invoke({"code": code, "period": period})
