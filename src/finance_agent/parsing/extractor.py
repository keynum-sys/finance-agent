"""LLM 结构化抽取:三大报表关键字段 -> Pydantic 模型。

第 4 周实现。设计要点:
- 按报表章节定向抽取(只喂对应章节的 chunk, 不喂全文 -> 准确率关键)
- 温度 0 + structured output; Pydantic 校验失败自动重试
- 会计恒等式校验: 资产 = 负债 + 权益, 不满足则标记 low_confidence 并重试
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BalanceSheet(BaseModel):
    total_assets: float = Field(..., description="资产总计(元)")
    total_liabilities: float = Field(..., description="负债合计(元)")
    total_equity: float = Field(..., description="所有者权益合计(元)")
    monetary_funds: float = Field(..., description="货币资金(元)")
    accounts_receivable: float = Field(..., description="应收账款(元)")
    inventory: float = Field(..., description="存货(元)")


class IncomeStatement(BaseModel):
    revenue: float = Field(..., description="营业收入(元)")
    operating_cost: float = Field(..., description="营业成本(元)")
    net_profit: float = Field(..., description="净利润(元)")
    net_profit_deducted: float = Field(..., description="扣非净利润(元)")
    rd_expense: float = Field(..., description="研发费用(元)")


class CashFlowStatement(BaseModel):
    operating_cash_flow: float = Field(..., description="经营活动现金流净额(元)")
    investing_cash_flow: float = Field(..., description="投资活动现金流净额(元)")
    financing_cash_flow: float = Field(..., description="筹资活动现金流净额(元)")


class ExtractedReport(BaseModel):
    """一份财报的抽取结果。"""

    source_pages: dict[str, int] = Field(
        default_factory=dict, description="字段组 -> 页码, 溯源用"
    )
    balance_sheet: BalanceSheet | None = None
    income_statement: IncomeStatement | None = None
    cash_flow_statement: CashFlowStatement | None = None
    confidence: Literal["high", "low"] = "high"

    def check_accounting_identity(self, tolerance: float = 0.005) -> bool:
        """会计恒等式校验: 资产 = 负债 + 权益(允许 0.5% 相对误差)。"""
        bs = self.balance_sheet
        if bs is None:
            return False
        diff = abs(bs.total_assets - bs.total_liabilities - bs.total_equity)
        return diff <= tolerance * bs.total_assets


def extract_report(chunks) -> ExtractedReport:
    """对章节 chunk 做 LLM 结构化抽取。

    TODO(第4周): langchain structured output + 重试 + 恒等式校验。
    """
    raise NotImplementedError
