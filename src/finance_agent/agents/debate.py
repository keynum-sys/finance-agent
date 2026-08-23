"""多智能体辩论节点: 多头分析师 vs 空头分析师 + 首席裁判。

选型理由: 单次 LLM 生成投资结论容易"一边倒"(立场漂移), 多空对抗式辩论
强制模型分别站在对立立场找证据, 再由裁判权衡, 结论更均衡、论据更扎实。

三轮对话结构(经典 debate 模式):

    多头立论(只给数据) -> 空头驳论(看得到多头论据) -> 裁决(看得到双方论据)

设计要点:
- 每轮 prompt 都附带数据摘要(报表关键字段 + 财务比率), 论据必须引用具体数字
- 裁决轮输出结构化 JSON({"stance", "verdict"}), 解析失败带错误反馈重试
  (与 extractor 相同的容错机制), 仍失败则降级: 保留多空论据, stance 记为"中性"
- chat_fn 注入, 单元测试全离线; 图节点层再兜底一层异常, 辩论失败不阻断主流程
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from finance_agent.config import settings
from finance_agent.parsing.extractor import ChatFn

Stance = Literal["看多", "看空", "中性"]


class DebateResult(BaseModel):
    """一场辩论的完整记录。"""

    bull_argument: str = Field(description="多头论据(条目列表文本)")
    bear_argument: str = Field(description="空头论据(条目列表文本)")
    stance: Stance = Field(description="裁决立场: 看多/看空/中性")
    verdict: str = Field(description="裁判综合结论(含关键数字引用)")


# --------------------------------------------------------------------------
# Prompt(纯文本, 便于审阅与迭代)
# --------------------------------------------------------------------------

_BULL_SYSTEM = """你是一名多头(看多)财务分析师。你的任务是站在乐观立场, 基于给定\
的财务数据给出最多 3 条看多论据。规则:
1. 每条论据必须引用具体数字和指标名称, 不允许"前景良好"这类空泛定性
2. 输出为条目列表, 每条一行, 以 "- " 开头, 不要额外解释
3. 数据不支持的论点不要强行编造"""

_BEAR_SYSTEM = """你是一名空头(看空)财务分析师, 正在与多头同事辩论。你的任务是针对\
多头的论据和给定数据给出最多 3 条看空或风险论据。规则:
1. 可以直接驳斥多头的具体论据, 也可以指出多头忽略的风险(如现金流质量、\
杠杆、可持续性)
2. 每条论据必须引用具体数字和指标名称
3. 输出为条目列表, 每条一行, 以 "- " 开头, 不要额外解释"""

_JUDGE_SYSTEM = """你是首席投资官, 负责裁决多空双方分析师的辩论。规则:
1. 客观权衡双方论据的数据支撑力度, 不预设立场, 也不各打五十大板
2. 只输出一个 JSON 对象, 不要任何解释、markdown 代码块或多余文本:
   {"stance": "看多", "verdict": "综合结论"}
3. stance 只能取 "看多"、"看空"、"中性" 三者之一
4. verdict 为 2-4 句中文结论: 先给核心判断, 再引用 1-2 个关键数字, \
最后点出最主要的风险"""


# --------------------------------------------------------------------------
# 数据摘要(纯函数)
# --------------------------------------------------------------------------


def _data_digest(extracted: dict, ratios: list[dict]) -> str:
    """把抽取结果和财务比率拼成辩论用数据摘要。

    三张报表只保留非 None 字段; 比率带解读。数据不足时返回占位提示,
    让辩论轮次仍可进行(图路由保证了 debate 之前 analyze 已产出比率)。
    """
    lines: list[str] = []
    for name in ("balance_sheet", "income_statement", "cash_flow_statement"):
        stmt = (extracted or {}).get(name) or {}
        items = [f"{k}={v:,.0f}" for k, v in stmt.items() if v is not None]
        if items:
            lines.append(f"{name}: " + ", ".join(items))
    for r in ratios or []:
        lines.append(f"比率 {r['name']}={r['value']:.2%}({r['interpretation']})")
    return "\n".join(lines) if lines else "(无可用财务数据)"


# --------------------------------------------------------------------------
# 辩论主流程
# --------------------------------------------------------------------------


def run_debate(
    extracted: dict,
    ratios: list[dict],
    chat_fn: ChatFn,
    max_retries: int | None = None,
) -> DebateResult:
    """执行三轮辩论, 返回结构化结果。

    extracted: ExtractedReport.model_dump()
    ratios:    [RatioResult.model_dump()]
    chat_fn:   LLM 调用函数(注入), 默认走 DeepSeek
    """
    if max_retries is None:
        max_retries = settings.extraction_max_retries
    digest = _data_digest(extracted, ratios)

    # 第一轮: 多头立论(只看数据)
    bull_argument = chat_fn([
        {"role": "system", "content": _BULL_SYSTEM},
        {"role": "user", "content": f"财务数据:\n{digest}\n\n请给出看多论据。"},
    ]).strip()

    # 第二轮: 空头驳论(看得到多头论据 -> 真正的对抗)
    bear_argument = chat_fn([
        {"role": "system", "content": _BEAR_SYSTEM},
        {"role": "user", "content": (
            f"财务数据:\n{digest}\n\n"
            f"多头分析师的论据:\n{bull_argument}\n\n"
            f"请驳斥并给出看空论据。"
        )},
    ]).strip()

    # 第三轮: 裁决(看得到双方论据), 结构化输出 + 失败重试
    judge_messages: list[dict] = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"财务数据:\n{digest}\n\n"
            f"多头论据:\n{bull_argument}\n\n"
            f"空头论据:\n{bear_argument}\n\n请裁决。"
        )},
    ]
    raw = ""
    for _ in range(max_retries):
        raw = chat_fn(judge_messages)
        try:
            data = json.loads(raw)
            return DebateResult(
                bull_argument=bull_argument,
                bear_argument=bear_argument,
                stance=data["stance"],
                verdict=data["verdict"],
            )
        except (json.JSONDecodeError, ValidationError, KeyError, ValueError) as e:
            # 与 extractor 相同的容错: 错误信息回传让模型自我纠正
            judge_messages.append({"role": "assistant", "content": raw})
            judge_messages.append({
                "role": "user",
                "content": f"你的输出有问题: {e}\n请重新输出 JSON, "
                           f"stance 只能取 看多/看空/中性。",
            })

    # 裁决始终无法解析: 降级保留多空论据(仍有信息量), 结论记原文
    return DebateResult(
        bull_argument=bull_argument,
        bear_argument=bear_argument,
        stance="中性",
        verdict=f"(裁决解析失败, 以下为原文) {raw[:200]}",
    )


# --------------------------------------------------------------------------
# 默认 LLM 实现(自然语言模式, 区别于 extractor 的 JSON mode)
# --------------------------------------------------------------------------


def default_chat(messages: list[dict]) -> str:
    """走 DeepSeek 的默认辩论调用。多空轮是自然语言, 裁决轮靠 prompt
    约束 + 重试保证 JSON, 不用 response_format(JSON mode 会强制多空轮
    也输出 JSON, 破坏条目列表格式)。"""
    from openai import OpenAI  # 延迟导入, 离线测试不触发

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,  # 辩论需要一点发散, 但仍偏确定性
    )
    return resp.choices[0].message.content or ""
