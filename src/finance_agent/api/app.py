"""FastAPI 服务(第 10 周): 把 LangGraph 流水线包装成 HTTP API。

端点:
- GET  /health   存活探测
- POST /analyze  {code, period, question?} -> 同步跑完整流水线, 返回完整报告
                 (含三表/比率/辩论/附注问答, 全流程一次约 1-3 分钟)
- POST /ask      {code, period, question} -> 纯 RAG 附注问答(带页码溯源),
                 不跑抽取/辩论, 需报告已入库(index)过

设计要点:
- create_app() 工厂: fetcher / chat_fn / store 可注入, 测试全离线
- 端点是普通 def: FastAPI 自动丢线程池, 不阻塞事件循环
- 流水线内部已有分级降级(下载失败/抽取失败 -> 仍出报告), API 层
  不再捕获——让调用方拿到 state.error 知道发生了什么
- /ask 直接复用向量库, 与 /analyze(question=...) 共享同一份索引
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from finance_agent.agents.graph import run_pipeline


class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="股票代码, 如 600519")
    period: str = Field(..., description="报告期, 如 2025-年报")
    question: str | None = Field(
        None, description="可选附注问题; 提供则启用 RAG 问答章节"
    )


class AskRequest(BaseModel):
    code: str = Field(..., description="股票代码")
    period: str = Field(..., description="报告期")
    question: str = Field(..., description="要问的附注问题")


def create_app(
    fetcher: Any | None = None,
    chat_fn: Any | None = None,
    store: Any | None = None,
) -> FastAPI:
    """构建 FastAPI 应用。依赖全部可注入, 单元测试离线。"""
    app = FastAPI(title="FinSight — A股财报分析智能体", version="0.2.0")

    _store = store  # None 时惰性创建真实向量库

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        """同步跑完整流水线, 返回最终 state(含 Markdown 报告)。

        耗时主要在 LLM 抽取(三表)+辩论(三轮), 约 1-3 分钟;
        PDF 有缓存则不重新下载。
        """
        state = run_pipeline(
            req.code,
            req.period,
            question=req.question,
            fetcher=fetcher,
            chat_fn=chat_fn,
            enable_rag=req.question is not None,
            store=_store,
        )
        return {
            "code": req.code,
            "period": req.period,
            "error": state.get("error"),
            "report_md": state.get("report_md", ""),
            "ratios": state.get("ratios") or [],
            "debate": state.get("debate"),
            "qa": state.get("qa"),
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        """纯 RAG 附注问答: 检索 + 带页码溯源生成。

        不跑抽取/辩论; 前提是该报告曾被 /analyze(question=...) 或
        index 节点入库过。检索不到时返回 found=false 而非报错。
        """
        if _store is not None:
            vector_store = _store
        else:
            from finance_agent.rag.store import ReportVectorStore

            vector_store = ReportVectorStore()  # 惰性: 首次调用才加载模型
        answer, citations = vector_store.query_with_citations(
            req.question, req.code, req.period, chat_fn=chat_fn
        )
        return {
            "code": req.code,
            "period": req.period,
            "question": req.question,
            "answer": answer,
            "citations": [
                {"page": c.page, "section": c.section, "snippet": c.snippet}
                for c in citations
            ],
        }

    return app


app = create_app()  # uvicorn finance_agent.api.app:app 的默认入口
