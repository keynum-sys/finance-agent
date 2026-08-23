"""财务比率计算与异常检测(纯规则, 无 LLM, 可单元测试——评测的基本盘)。"""

from __future__ import annotations

from dataclasses import dataclass

from finance_agent.config import settings
from finance_agent.parsing.extractor import ExtractedReport


@dataclass
class RatioResult:
    name: str
    value: float
    interpretation: str
    source_page: int | None = None


def compute_ratios(report: ExtractedReport) -> list[RatioResult]:
    """计算核心财务比率。缺数据时跳过对应比率。"""
    results: list[RatioResult] = []
    bs, ic, cf = report.balance_sheet, report.income_statement, report.cash_flow_statement

    if bs and bs.total_equity:
        if ic and ic.net_profit:
            roe = ic.net_profit / bs.total_equity
            results.append(RatioResult("净资产收益率 ROE", roe, _interp_roe(roe)))
    if bs and bs.total_assets:
        debt_ratio = bs.total_liabilities / bs.total_assets
        results.append(
            RatioResult("资产负债率", debt_ratio, _interp_debt(debt_ratio))
        )
    if ic and ic.revenue:
        gross_margin = (ic.revenue - ic.operating_cost) / ic.revenue
        results.append(
            RatioResult("毛利率", gross_margin, "毛利率水平需结合行业均值对比")
        )
    if ic and ic.net_profit and cf and cf.operating_cash_flow:
        ocf_np = cf.operating_cash_flow / ic.net_profit
        interp = (
            "经营现金流对净利润覆盖充分, 盈利质量较好"
            if ocf_np >= 1.0
            else "经营现金流未能覆盖净利润, 需关注盈利质量(应收/存货占款)"
        )
        results.append(RatioResult("经营现金流/净利润", ocf_np, interp))
    return results


@dataclass
class Anomaly:
    metric: str
    current: float
    previous: float
    change_rate: float
    note: str


def detect_anomalies(current: ExtractedReport, previous: ExtractedReport) -> list[Anomaly]:
    """同比异常检测: 关键指标变动超过阈值即预警。第 7 周扩展为规则+LLM双通道。"""
    threshold = settings.yoy_anomaly_threshold
    pairs = [
        ("营业收入", _get(current, "revenue"), _get(previous, "revenue")),
        ("净利润", _get(current, "net_profit"), _get(previous, "net_profit")),
        ("应收账款", _get(current, "accounts_receivable"), _get(previous, "accounts_receivable")),
        ("存货", _get(current, "inventory"), _get(previous, "inventory")),
    ]
    anomalies: list[Anomaly] = []
    for name, cur, prev in pairs:
        if cur is None or prev is None or prev == 0:
            continue
        change = (cur - prev) / abs(prev)
        if abs(change) >= threshold:
            anomalies.append(
                Anomaly(name, cur, prev, change, f"{name}同比变动 {change:+.1%}, 超过 ±{threshold:.0%} 阈值")
            )
    return anomalies


def _get(report: ExtractedReport, field: str) -> float | None:
    ic, bs = report.income_statement, report.balance_sheet
    if ic and field in type(ic).model_fields:
        return getattr(ic, field)
    if bs and field in type(bs).model_fields:
        return getattr(bs, field)
    return None


def _interp_roe(roe: float) -> str:
    if roe >= 0.15:
        return "ROE >= 15%, 盈利能力较强"
    if roe >= 0.08:
        return "ROE 处于 8%-15%, 中等水平"
    return "ROE < 8%, 盈利能力偏弱(注意区分亏损与低盈利)"


def _interp_debt(ratio: float) -> str:
    if ratio >= 0.7:
        return "资产负债率 >= 70%, 杠杆偏高(金融地产行业需另行标准)"
    if ratio >= 0.4:
        return "资产负债率 40%-70%, 常规区间"
    return "资产负债率 < 40%, 杠杆偏低"
