# -*- coding: utf-8 -*-
"""eval 评分函数单元测试: 纯离线, 验证字段级 F1 的四种判定与汇总。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.eval.run_eval import aggregate, group_by_industry, load_golden, score_report

# ---------------------------------------------------------------- 测试数据

_GOLD = {
    "code": "000001", "period": "2026-一季报", "industry": "银行",
    "balance_sheet": {"total_assets": 6033962000000.0, "monetary_funds": None},
    "income_statement": {"revenue": 35277000000.0, "net_profit_deducted": None},
    "cash_flow_statement": {"operating_cash_flow": 37802000000.0},
}


def test_perfect_prediction():
    pred = {
        "balance_sheet": {"total_assets": 6033962000000.0, "monetary_funds": None},
        "income_statement": {"revenue": 35277000000.0, "net_profit_deducted": None},
        "cash_flow_statement": {"operating_cash_flow": 37802000000.0},
    }
    r = score_report(_GOLD, pred)
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["errors"] == []
    # TP=3: 三个非空字段全对; 双 null(正确拒绝)不计入


def test_within_tolerance_counts_as_hit():
    pred = {
        "balance_sheet": {"total_assets": 6033962000000.0 * 1.004},  # 0.4% 误差
        "income_statement": {"revenue": 35277000000.0, "net_profit_deducted": None},
        "cash_flow_statement": {"operating_cash_flow": 37802000000.0},
        # monetary_funds 缺失 = 预测 null
    }
    r = score_report(_GOLD, pred)
    assert r["tp"] == 3
    assert r["precision"] == 1.0                     # 没有错误答案
    assert r["recall"] == 1.0                        # null 对 null 是正确拒绝, 不是漏抽


def test_miss_counts_as_fn():
    pred = {
        "balance_sheet": {"total_assets": 6033962000000.0, "monetary_funds": None},
        "income_statement": {"revenue": None, "net_profit_deducted": None},  # 漏抽
        "cash_flow_statement": {"operating_cash_flow": 37802000000.0},
    }
    r = score_report(_GOLD, pred)
    assert r["fn"] == 1
    assert r["recall"] == 2 / 3
    assert r["errors"][0]["kind"] == "漏抽"


def test_wrong_value_counts_as_fp_and_fn():
    pred = {
        "balance_sheet": {"total_assets": 5951722000000.0,   # 抽成了母公司数
                          "monetary_funds": None},
        "income_statement": {"revenue": 35277000000.0, "net_profit_deducted": None},
        "cash_flow_statement": {"operating_cash_flow": 37802000000.0},
    }
    r = score_report(_GOLD, pred)
    assert r["fp"] == 1 and r["fn"] == 1              # 抽错 = 双重惩罚
    assert r["errors"][0]["kind"] == "数值错误"
    assert r["tp"] == 2                               # revenue + ocf
    assert r["precision"] == 2 / 3
    assert r["recall"] == 2 / 3


def test_hallucination_on_null_field_counts_as_fp():
    pred = {
        "balance_sheet": {"total_assets": 6033962000000.0,
                          "monetary_funds": 260145000000.0},  # 银行无此科目, 编了数
        "income_statement": {"revenue": 35277000000.0, "net_profit_deducted": None},
        "cash_flow_statement": {"operating_cash_flow": 37802000000.0},
    }
    r = score_report(_GOLD, pred)
    assert r["fp"] == 1
    assert r["precision"] == 3 / 4
    assert r["errors"][0]["kind"].startswith("幻觉")


def test_unit_error_detected():
    """单位换算错误(百万 vs 元, 差 100 倍)必须判错。"""
    pred = {
        "balance_sheet": {"total_assets": 60339620000.0, "monetary_funds": None},
        "income_statement": {"revenue": 35277000000.0, "net_profit_deducted": None},
        "cash_flow_statement": {"operating_cash_flow": 37802000000.0},
    }
    r = score_report(_GOLD, pred)
    assert r["fn"] == 1


def test_whole_statement_missing_counts_all_fn():
    pred = {"balance_sheet": None,                    # 整表抽取失败
            "income_statement": {"revenue": 35277000000.0, "net_profit_deducted": None},
            "cash_flow_statement": {"operating_cash_flow": 37802000000.0}}
    r = score_report(_GOLD, pred)
    assert r["fn"] == 1                               # 只有 total_assets 一个非空金标


def test_negative_gold_matched():
    gold = {"cash_flow_statement": {"financing_cash_flow": -35634000000.0}}
    pred = {"cash_flow_statement": {"financing_cash_flow": -35634000000.0}}
    r = score_report(gold, pred)
    assert r["tp"] == 1


# ---------------------------------------------------------------- 汇总

def test_aggregate_micro_average():
    r1 = {"tp": 3, "fp": 1, "fn": 0}
    r2 = {"tp": 1, "fp": 0, "fn": 1}
    m = aggregate([r1, r2])
    assert m["tp"] == 4 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == 4 / 5
    assert m["recall"] == 4 / 5
    assert abs(m["f1"] - 0.8) < 1e-9


def test_group_by_industry():
    rows = [
        {"golden": {"industry": "银行"}, "result": {"tp": 3, "fp": 0, "fn": 1}},
        {"golden": {"industry": "银行"}, "result": {"tp": 2, "fp": 1, "fn": 0}},
        {"golden": {"industry": "白酒"}, "result": {"tp": 5, "fp": 1, "fn": 1}},
    ]
    by = group_by_industry(rows)
    assert by["银行"]["tp"] == 5 and by["银行"]["fp"] == 1 and by["银行"]["fn"] == 1
    assert by["白酒"]["tp"] == 5


# ---------------------------------------------------------------- 金标文件

def test_golden_file_loadable_and_identity():
    """金标文件本身要能加载, 且茅台资产负债表满足会计恒等式。"""
    golden = load_golden()
    assert len(golden) == 2
    by_code = {g["code"]: g for g in golden}
    mt = by_code["600519"]
    bs = mt["balance_sheet"]
    assert abs(bs["total_assets"] - bs["total_liabilities"] - bs["total_equity"]) \
        <= 0.005 * bs["total_assets"]                # 恒等式(0.5% 容忍)
    bank = by_code["000001"]
    assert bank["balance_sheet"]["inventory"] is None  # 银行股 null 案例
    # 银行金标单位已换算为元(百万元 * 1e6)
    assert bank["balance_sheet"]["total_assets"] == 6033962000000.0
