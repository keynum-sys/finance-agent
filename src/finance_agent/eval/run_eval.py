"""自动化评测: 字段级抽取 F1 + 分行业统计。

评测集格式(eval/golden.jsonl, 每行一份财报, 数值单位一律"元"):
{"code": "600519", "period": "2025-年报", "industry": "白酒",
 "balance_sheet": {"total_assets": 303834844021.44, ...},
 "income_statement": {...}, "cash_flow_statement": {...}}

金标建法: 从财报 PDF 导出三表原文, 人工核对期末数后录入(不经过 LLM,
保证独立性)。null 表示该报表确实没有此科目(如银行股没有存货)。

字段判定(数值相对误差 <= 0.5% 判对):
- gold 非空, 预测匹配            -> TP
- gold 非空, 预测为 null          -> FN(漏抽)
- gold 非空, 预测不匹配           -> 1 FP + 1 FN(抽错: 既没抽对, 又交了错误答案)
- gold 为 null, 预测也为 null     -> 正确拒绝, 不计入 P/R
- gold 为 null, 预测非空          -> FP(幻觉: 报表里不存在的科目编了数)

指标: micro 精确率/召回率/F1(全部字段合并算), 分行业统计 -> 暴露
LLM 弱点(如银行股单位换算、母公司/合并报表混淆), 面试谈资。
"""

from __future__ import annotations

import json
from pathlib import Path

from finance_agent.config import settings

STATEMENTS = ("balance_sheet", "income_statement", "cash_flow_statement")


def load_golden(path: Path | None = None) -> list[dict]:
    path = path or Path(__file__).resolve().parents[3] / "eval" / "golden.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# 单份财报的字段级评分(纯函数)
# --------------------------------------------------------------------------


def _match(gv, pv, tolerance: float) -> bool:
    """数值: 相对误差容忍; 其他类型: 严格相等。"""
    if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
        return abs(gv - pv) <= tolerance * max(abs(gv), 1e-9)
    return gv == pv


def score_report(
    golden: dict, predicted: dict, tolerance: float | None = None
) -> dict:
    """单份财报的字段级评分。

    返回 {tp, fp, fn, precision, recall, f1, errors: [{field, gold, pred, kind}]}
    predicted: ExtractedReport.model_dump()(允许整表为 None)
    """
    tolerance = settings.numeric_tolerance if tolerance is None else tolerance
    tp = fp = fn = 0
    errors: list[dict] = []
    for stmt in STATEMENTS:
        g = golden.get(stmt) or {}
        p = predicted.get(stmt) or {}   # 整表抽取失败 -> 全部按 null 处理
        for k, gv in g.items():
            pv = p.get(k)
            if gv is None:
                if pv is not None:      # 幻觉: 科目不存在却编了数
                    fp += 1
                    errors.append({"field": f"{stmt}.{k}", "gold": None,
                                   "pred": pv, "kind": "幻觉(科目不存在)"})
                continue                # 双 null: 正确拒绝, 不计
            if pv is None:              # 漏抽
                fn += 1
                errors.append({"field": f"{stmt}.{k}", "gold": gv,
                               "pred": None, "kind": "漏抽"})
            elif _match(gv, pv, tolerance):
                tp += 1
            else:                       # 抽错: 交了错误答案且没抽对
                fp += 1
                fn += 1
                errors.append({"field": f"{stmt}.{k}", "gold": gv,
                               "pred": pv, "kind": "数值错误"})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "errors": errors}


# --------------------------------------------------------------------------
# 汇总(多份财报 micro 平均 + 分行业)
# --------------------------------------------------------------------------


def aggregate(results: list[dict]) -> dict:
    """micro 汇总: 把所有 TP/FP/FN 加总后算 P/R/F1。"""
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def group_by_industry(rows: list[dict]) -> dict[str, dict]:
    """按行业分组汇总。rows: [{golden, result}, ...]"""
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["golden"].get("industry", "未知"), []).append(row["result"])
    return {ind: aggregate(rs) for ind, rs in out.items()}


# --------------------------------------------------------------------------
# Runner: 金标 -> 抽取 -> 评分 -> 报告
# --------------------------------------------------------------------------


def evaluate_report(golden: dict, chat_fn=None) -> dict:
    """对一份金标财报跑完整抽取并评分(PDF 走缓存)。"""
    from finance_agent.data.fetcher import ReportFetcher
    from finance_agent.parsing.extractor import extract_report

    pdf_path = ReportFetcher().download(golden["code"], golden["period"])
    predicted = extract_report(str(pdf_path), chat_fn=chat_fn).model_dump()
    return score_report(golden, predicted)


def run_eval(chat_fn=None, golden_path: Path | None = None) -> dict:
    """跑全量评测集: 抽取 -> 评分 -> 总体 + 分行业报告。

    返回 {overall, by_industry, rows}; 同时打印人读报告。
    """
    golden_list = load_golden(golden_path)
    rows = []
    print(f"评测集: {len(golden_list)} 份财报")
    for golden in golden_list:
        result = evaluate_report(golden, chat_fn=chat_fn)
        rows.append({"golden": golden, "result": result})
        tag = f"{golden['code']} {golden['period']}"
        print(f"  {tag}: P={result['precision']:.2%} R={result['recall']:.2%} "
              f"F1={result['f1']:.2%} (TP={result['tp']} FP={result['fp']} FN={result['fn']})")
        for e in result["errors"]:
            print(f"    [{e['kind']}] {e['field']}: gold={e['gold']} pred={e['pred']}")

    overall = aggregate([r["result"] for r in rows])
    by_industry = group_by_industry(rows)
    print("-" * 60)
    print(f"总体: P={overall['precision']:.2%} R={overall['recall']:.2%} "
          f"F1={overall['f1']:.2%} (TP={overall['tp']} FP={overall['fp']} FN={overall['fn']})")
    for ind, m in sorted(by_industry.items()):
        print(f"  {ind}: F1={m['f1']:.2%} (P={m['precision']:.2%} R={m['recall']:.2%})")
    return {"overall": overall, "by_industry": by_industry, "rows": rows}


if __name__ == "__main__":
    run_eval()
