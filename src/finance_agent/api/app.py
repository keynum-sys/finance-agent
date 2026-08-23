"""FastAPI 服务(第 9 周实现)。

端点规划:
- POST /analyze   {code, period} -> 触发完整流水线, 返回报告(异步任务)
- GET  /report    查询报告结果 + 引用
- POST /ask       财报 RAG 问答(带溯源)
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="FinSight — A股财报分析智能体", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(code: str, period: str) -> dict:
    """触发分析流水线。TODO(第9周): 接入 LangGraph + 后台任务。"""
    raise NotImplementedError
