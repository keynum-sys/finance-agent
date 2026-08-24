"""LangGraph 流水线编排: download -> extract -> analyze -> {debate, index} -> [rag_qa] -> report。

选型理由: 财报分析是确定性流程, 图编排比 ReAct 自由循环更可控、可回溯。

图结构(条件边处理失败分支; rag 部分需 enable_rag=True 才装配):

    START -> download
    download --成功--> extract --抽取到数据--> analyze --+--> debate --有question--> rag_qa -> report -> END
        |                |                              |        |                            ^
        +--失败----------+--全部失败--> report <--------+--------+--无question-----------------+
                                                          |
                                                          +--> index(与 debate 并行) ------> report

设计要点:
- 每个节点是一个纯函数: 读 state -> 返回部分更新(dict), 不抛异常
  (节点内异常转成 state["error"], 由条件边路由到 report 收尾)
- debate 是增强节点: 多头/空头/裁判三轮辩论, 失败静默跳过, 不阻断主流程
- index 是 RAG 增强节点: 报告章节子块入向量库(与 debate 并行扇出),
  失败静默跳过; rag_qa 仅在 state 带 question 时被路由到, 问答失败
  也不阻断主流程(错误写进 qa.answer)
- fetcher / chat_fn / store 依赖注入, 单元测试全离线;
  enable_rag=False(默认)时不装配 RAG 节点, 零开销
- report 节点为确定性模板(数据必须可机器校验, 叙述性结论交给 debate/rag_qa)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from finance_agent.agents import debate as debate_mod
from finance_agent.analysis.ratios import compute_ratios
from finance_agent.data.fetcher import ReportFetcher
from finance_agent.parsing.extractor import ChatFn, ExtractedReport, extract_report


class ReportState(TypedDict, total=False):
    """流水线共享状态。total=False: 所有键可选, 节点按需写入。"""

    # 输入
    code: str                   # 股票代码, 如 "600519"
    period: str                 # 报告期, 如 "2025-年报"
    question: str               # 可选: 附注问题(存在时路由到 rag_qa)
    # 中间产物
    pdf_path: str
    extracted: dict             # ExtractedReport.model_dump()
    ratios: list[dict]          # [RatioResult.model_dump()]
    debate: dict                # DebateResult.model_dump()(可选, 失败时缺失)
    indexed: bool               # index 节点成功入库的标记
    indexed_count: int          # 入库子块数
    qa: dict                    # {"question","answer","citations":[{page,section,snippet}]}
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


def _make_debate_node(chat_fn: ChatFn | None):
    def debate(state: ReportState) -> dict:
        """多空辩论(增强节点): 失败静默跳过, 绝不阻断主流程。

        走到这里说明 analyze 已产出 ratios, 数据一定存在;
        可能失败的是 LLM 调用本身(网络/解析), 由 try 兜底。
        """
        chat = chat_fn or debate_mod.default_chat
        try:
            result = debate_mod.run_debate(
                state.get("extracted") or {}, state.get("ratios") or [], chat
            )
        except Exception:
            return {}  # 辩论是锦上添花, 失败就出无辩论版报告
        return {"debate": result.model_dump()}

    return debate


def _make_index_node(store_factory):
    def index(state: ReportState) -> dict:
        """报告章节子块入向量库(RAG 增强节点, 与 debate 并行)。

        只依赖 pdf_path, 失败静默跳过——入库失败最多意味着
        本次附注问答检索不到, 不影响主分析流程。
        """
        if state.get("error") or not state.get("pdf_path"):
            return {}
        try:
            from finance_agent.parsing.pdf_parser import extract_pages, split_into_chunks

            chunks = split_into_chunks(extract_pages(state["pdf_path"]))
            n = store_factory().add_chunks(state["code"], state["period"], chunks)
        except Exception:
            return {}
        return {"indexed": True, "indexed_count": n}

    return index


def _make_rag_node(store_factory, chat_fn: ChatFn | None):
    def rag_qa(state: ReportState) -> dict:
        """附注问答: 向量检索 + 带页码溯源的生成。

        仅当 state 带 question 时被路由到; 问答失败不阻断主流程,
        错误信息写进 qa.answer(用户可见但不影响其他章节)。
        """
        question = state.get("question")
        if not question:
            return {}
        try:
            answer, citations = store_factory().query_with_citations(
                question, state["code"], state["period"], chat_fn=chat_fn
            )
            cits = [
                {"page": c.page, "section": c.section, "snippet": c.snippet}
                for c in citations
            ]
        except Exception as e:
            return {"qa": {
                "question": question,
                "answer": f"(附注问答暂不可用: {e})",
                "citations": [],
            }}
        return {"qa": {"question": question, "answer": answer, "citations": cits}}

    return rag_qa


def report(state: ReportState) -> dict:
    """生成最终 Markdown 报告: 确定性模板 + 辩论结论(若有)。"""
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

    d = state.get("debate")
    if d:
        lines += [
            "## 多空辩论", "",
            "### 多头论据", "", d["bull_argument"], "",
            "### 空头论据", "", d["bear_argument"], "",
            f"### 裁决({d['stance']})", "", d["verdict"], "",
        ]

    qa = state.get("qa")
    if qa:
        lines += [
            "## 附注问答", "",
            f"**问**: {qa['question']}", "",
            f"**答**: {qa['answer']}", "",
        ]
        if qa["citations"]:
            lines += ["**引用来源**:", ""]
            for c in qa["citations"]:
                lines.append(f"- 第{c['page']}页({c['section']}): {c['snippet']}")
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


def _route_after_debate(state: ReportState) -> str:
    """带附注问题 -> 问答节点; 否则直接出报告。"""
    return "rag_qa" if state.get("question") else "report"


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------


def build_graph(
    fetcher: ReportFetcher | None = None,
    chat_fn: ChatFn | None = None,
    enable_rag: bool = False,
    store=None,
) -> object:
    """构建并编译流水线图。

    fetcher / chat_fn 均可注入假实现, 测试全离线。
    enable_rag=True 时装配 index / rag_qa 两个节点(见模块 docstring 图);
    store 注入 ReportVectorStore 兼容对象(有 add_chunks / query_with_citations),
    缺省时惰性创建真实 Chroma 向量库(仅在节点真正运行时才 import/加载模型)。
    返回编译后的图, 用 graph.invoke({"code": ..., "period": ..., "question": ...}) 运行。
    """
    fetcher = fetcher or ReportFetcher()

    # RAG 向量库惰性创建: 关闭时零开销, 开启时也避免在 import 阶段加载模型
    _store_holder: list = []

    def get_store():
        if not _store_holder:
            if store is None:
                from finance_agent.rag.store import ReportVectorStore

                _store_holder.append(ReportVectorStore())
            else:
                _store_holder.append(store)
        return _store_holder[0]

    g = StateGraph(ReportState)
    g.add_node("download", _make_download_node(fetcher))
    g.add_node("extract", _make_extract_node(chat_fn))
    g.add_node("analyze", analyze)
    g.add_node("debate", _make_debate_node(chat_fn))
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
    g.add_edge("analyze", "debate")
    if enable_rag:
        g.add_node("index", _make_index_node(get_store))
        g.add_node("rag_qa", _make_rag_node(get_store, chat_fn))
        # index 与 debate 从 analyze 扇出并行; 都完成后汇入 report
        g.add_edge("analyze", "index")
        g.add_edge("index", "report")
        g.add_conditional_edges(
            "debate",
            _route_after_debate,
            {"rag_qa": "rag_qa", "report": "report"},
        )
        g.add_edge("rag_qa", "report")
    else:
        g.add_edge("debate", "report")
    g.add_edge("report", END)

    return g.compile()


def run_pipeline(code: str, period: str, question: str | None = None, **kwargs) -> ReportState:
    """便捷入口: 跑完整流水线, 返回最终 state(含 report_md)。

    传 question 时自动启用 RAG 并触发附注问答(页码溯源)。
    """
    if question:
        kwargs.setdefault("enable_rag", True)
    graph = build_graph(**kwargs)
    state: ReportState = {"code": code, "period": period}
    if question:
        state["question"] = question
    return graph.invoke(state)
