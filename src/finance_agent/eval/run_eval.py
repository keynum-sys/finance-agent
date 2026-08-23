"""自动化评测: 字段级抽取准确率 + 分行业统计。

评测集格式(eval/golden.jsonl, 每行一份财报):
{"code": "000001", "period": "2025-年报", "industry": "银行",
 "balance_sheet": {"total_assets": 12000000000000.0, ...},
 "income_statement": {...}, "cash_flow_statement": {...}}

指标:
- 字段级 precision / recall(数值相对误差 <= 0.5% 判对)
- 分行业统计 -> 暴露 LLM 弱点(面试谈资)
"""

from __future__ import annotations

import json
from pathlib import Path

from finance_agent.config import settings


def load_golden(path: Path | None = None) -> list[dict]:
    path = path or Path("eval/golden.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def field_level_accuracy(golden: dict, predicted: dict, tolerance: float | None = None) -> dict:
    """单份财报的字段级评测。返回 {precision, recall, wrong_fields}。"""
    tolerance = tolerance or settings.numeric_tolerance
    wrong: list[str] = []
    hit = 0
    total_gold = 0
    total_pred = 0
    for stmt in ("balance_sheet", "income_statement", "cash_flow_statement"):
        g, p = golden.get(stmt) or {}, predicted.get(stmt) or {}
        total_gold += len(g)
        total_pred += len(p)
        for k, gv in g.items():
            if k not in p:
                wrong.append(f"{stmt}.{k}:missing")
                continue
            pv = p[k]
            if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
                ok = abs(gv - pv) <= tolerance * max(abs(gv), 1e-9)
            else:
                ok = gv == pv
            hit += ok
            if not ok:
                wrong.append(f"{stmt}.{k}")
    precision = hit / total_pred if total_pred else 0.0
    recall = hit / total_gold if total_gold else 0.0
    return {"precision": precision, "recall": recall, "wrong_fields": wrong}


def run_eval() -> None:
    """跑全量评测集, 输出总体 + 分行业准确率。TODO: 接入抽取流水线后完成。"""
    golden = load_golden()
    # TODO(第6周): for each -> 调用流水线抽取 -> field_level_accuracy -> 汇总报告
    print(f"loaded {len(golden)} golden reports")
