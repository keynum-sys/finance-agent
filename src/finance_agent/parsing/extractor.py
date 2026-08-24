"""LLM 结构化抽取:三大报表关键字段 -> Pydantic 模型。

设计要点:
- 按报表章节定向抽取(只喂对应报表的文本, 不喂全文 -> 准确率关键)
- 温度 0 + JSON mode; Pydantic 校验失败把错误信息回传给模型自动重试
- 会计恒等式校验: 资产 = 负债 + 权益, 不满足则重试, 仍失败标记 low confidence
- LLM 调用通过 chat_fn 参数注入, 单元测试不需要真实 API(离线可测)

字段全部允许为 None 的原因:
- 银行/券商报表没有"存货""应收账款"等科目
- 扣非净利润通常不在合并利润表正文, 而在"主要财务指标"章节
下游(ratios.py)已对 None 做了防御, 缺哪个就跳过哪个指标。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from finance_agent.config import settings
from finance_agent.parsing.pdf_parser import (
    extract_pages,
    locate_statements,
    statement_text_from_pages,
)

# LLM 调用签名: 输入 messages, 返回模型回复文本。测试时注入假实现。
ChatFn = Callable[[list[dict]], str]

# --------------------------------------------------------------------------
# Pydantic 数据模型(抽取目标)
# --------------------------------------------------------------------------


class BalanceSheet(BaseModel):
    total_assets: float | None = Field(None, description="资产总计(元)")
    total_liabilities: float | None = Field(None, description="负债合计(元)")
    total_equity: float | None = Field(None, description="所有者权益合计(元)")
    monetary_funds: float | None = Field(None, description="货币资金(元)")
    accounts_receivable: float | None = Field(None, description="应收账款(元)")
    inventory: float | None = Field(None, description="存货(元)")


class IncomeStatement(BaseModel):
    revenue: float | None = Field(None, description="营业收入(元)")
    operating_cost: float | None = Field(None, description="营业成本(元)")
    net_profit: float | None = Field(None, description="净利润(元)")
    net_profit_deducted: float | None = Field(None, description="扣非净利润(元)")
    rd_expense: float | None = Field(None, description="研发费用(元)")


class CashFlowStatement(BaseModel):
    operating_cash_flow: float | None = Field(None, description="经营活动现金流净额(元)")
    investing_cash_flow: float | None = Field(None, description="投资活动现金流净额(元)")
    financing_cash_flow: float | None = Field(None, description="筹资活动现金流净额(元)")


class ExtractedReport(BaseModel):
    """一份财报的抽取结果。"""

    source_pages: dict[str, int] = Field(
        default_factory=dict, description="报表名 -> 起始页码, 溯源用"
    )
    balance_sheet: BalanceSheet | None = None
    income_statement: IncomeStatement | None = None
    cash_flow_statement: CashFlowStatement | None = None
    confidence: Literal["high", "low"] = "high"

    def check_accounting_identity(self, tolerance: float | None = None) -> bool:
        """会计恒等式校验: 资产 = 负债 + 权益(默认允许 0.5% 相对误差)。"""
        if tolerance is None:
            tolerance = settings.numeric_tolerance
        bs = self.balance_sheet
        if bs is None or bs.total_assets is None:
            return False
        if bs.total_liabilities is None or bs.total_equity is None:
            return False
        diff = abs(bs.total_assets - bs.total_liabilities - bs.total_equity)
        return diff <= tolerance * abs(bs.total_assets)


# --------------------------------------------------------------------------
# Prompt 构造(纯函数)
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是一名严谨的财务数据抽取引擎。你的唯一任务是把中国 A 股财报中的\
报表文本抽取为 JSON, 规则:
1. 只输出一个 JSON 对象, 不要任何解释、markdown 代码块或多余文本
2. 数值单位统一为"元": 报表里的"千元/百万元/亿元"必须换算成元
3. 数字用纯数字表示: 不要千分位逗号, 负数用负号, 括号表示的负数要去括号加负号
4. 只抽取期末余额(资产负债表)或本期发生额(利润表/现金流量表), 不要上期数
5. 报表中确实不存在的科目(如银行股没有存货), 对应字段填 null"""


def _schema_prompt(model_cls: type[BaseModel]) -> str:
    """从 Pydantic 模型自动生成字段说明, 模型改字段时 prompt 自动同步。"""
    lines = []
    for name, field in model_cls.model_fields.items():
        lines.append(f'  "{name}": 数值或 null  // {field.description}')
    return "\n".join(lines)


def _user_prompt(statement_name: str, text: str, model_cls: type[BaseModel]) -> str:
    unit_hint = _unit_hint(text)
    return (
        f"请从下面的《{statement_name}》文本中抽取字段, 输出 JSON:\n"
        f"{{\n{_schema_prompt(model_cls)}\n}}\n"
        + (f"注意: 本报表标注的单位为「{unit_hint}」, "
           f"输出前必须把每个数值换算成元后再填写。\n\n" if unit_hint else "")
        + f"----- {statement_name} 文本开始 -----\n"
        f"{text}\n"
        f"----- {statement_name} 文本结束 -----"
    )


# --------------------------------------------------------------------------
# 单位检测(纯函数): 评测发现银行股"百万元"换算易错 10 倍, 显式提示给 LLM
# --------------------------------------------------------------------------

_UNIT_PATTERN = re.compile(r"单位[：:]\s*(?:人民币)?(百万元|亿元|千元|万元|元)")


def detect_unit(text: str) -> str | None:
    """从报表文本头部检测标注的货币单位, 如"货币单位：人民币百万元"。"""
    m = _UNIT_PATTERN.search(text[:500])
    return m.group(1) if m else None


def _unit_hint(text: str) -> str | None:
    unit = detect_unit(text)
    return unit if unit and unit != "元" else None


# --------------------------------------------------------------------------
# LLM 调用(默认实现, 走 DeepSeek)
# --------------------------------------------------------------------------


def _default_chat(messages: list[dict]) -> str:
    from openai import OpenAI  # 延迟导入, 测试时无需安装环境也能跑纯函数测试

    client = OpenAI(
        api_key=settings.llm_api_key, base_url=settings.llm_base_url
    )
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},  # DeepSeek JSON mode
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------
# 单张报表抽取(带重试)
# --------------------------------------------------------------------------


def _has_any_value(model: BaseModel) -> bool:
    """全 null 结果视为抽取失败(比如模型没找到任何科目)。"""
    return any(getattr(model, name) is not None for name in type(model).model_fields)


def _extract_statement(
    text: str,
    model_cls: type[BaseModel],
    statement_name: str,
    chat_fn: ChatFn,
    max_retries: int | None = None,
    extra_check: Callable[[BaseModel], str | None] | None = None,
) -> BaseModel | None:
    """抽取单张报表, 校验失败把错误反馈给模型重试。

    extra_check: 额外的业务校验(如会计恒等式), 返回错误描述或 None(通过)。
    """
    if max_retries is None:
        max_retries = settings.extraction_max_retries
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(statement_name, text, model_cls)},
    ]
    for _ in range(max_retries):
        raw = chat_fn(messages)
        try:
            data = json.loads(raw)
            model = model_cls.model_validate(data)
            if not _has_any_value(model):
                raise ValueError("所有字段都是 null, 未抽取到任何科目")
            error = extra_check(model) if extra_check else None
            if error:
                raise ValueError(error)
            return model
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            # 把错误信息回传, 让模型自我纠正(核心容错机制)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"你的输出有问题: {e}\n请严格按规则重新输出 JSON, "
                    f"不要任何解释或代码块。",
                }
            )
    return None


def _identity_check(bs: BalanceSheet) -> str | None:
    """会计恒等式校验, 作为资产负债表抽取的额外校验。"""
    if not ExtractedReport(balance_sheet=bs).check_accounting_identity():
        return (
            f"会计恒等式不成立: 资产总计({bs.total_assets}) != "
            f"负债合计({bs.total_liabilities}) + 所有者权益合计({bs.total_equity})。"
            f"请检查是否抄错数字、是否抽成了上期数或母公司数。"
        )
    return None


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

# (报表名, 模型, 报告字段, 额外校验)
_STATEMENT_SPECS: list[tuple[str, type[BaseModel], str, Callable | None]] = [
    ("合并资产负债表", BalanceSheet, "balance_sheet", _identity_check),
    ("合并利润表", IncomeStatement, "income_statement", None),
    ("合并现金流量表", CashFlowStatement, "cash_flow_statement", None),
]


def extract_report(pdf_path: str, chat_fn: ChatFn | None = None) -> ExtractedReport:
    """解析 PDF 并抽取三大报表关键字段。

    pdf_path: 财报 PDF 路径
    chat_fn:  LLM 调用函数, 默认走 DeepSeek; 测试时注入假实现
    """
    chat_fn = chat_fn or _default_chat
    pages = extract_pages(pdf_path)
    locs = locate_statements(pages)

    report = ExtractedReport(
        source_pages={name: start for name, (start, _end) in locs.items()}
    )
    for statement_name, model_cls, field, check in _STATEMENT_SPECS:
        if statement_name not in locs:
            continue  # 未定位到该报表(如季报无现金流量表正文), 对应字段保持 None
        text = statement_text_from_pages(pages, statement_name)
        model = _extract_statement(text, model_cls, statement_name, chat_fn, extra_check=check)
        setattr(report, field, model)

    # 置信度: 三表齐全且恒等式通过才算 high
    all_extracted = all(
        getattr(report, field) is not None for _, _, field, _ in _STATEMENT_SPECS
    )
    if not all_extracted or not report.check_accounting_identity():
        report.confidence = "low"
    return report
