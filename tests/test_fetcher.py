"""fetcher 单元测试（不联网，只测纯逻辑）。"""

from finance_agent.data.fetcher import Announcement, ReportFetcher, pick_best_announcement


def _ann(title: str, t: str = "2026-04-01") -> Announcement:
    return Announcement(title=title, pdf_url=f"http://example.com/{title}.pdf", announcement_time=t)


def test_pick_best_prefers_clean_title():
    anns = [
        _ann("平安银行：2025年年度报告摘要"),
        _ann("平安银行：2025年年度报告（更新后）"),
        _ann("平安银行：2025年年度报告"),
    ]
    best = pick_best_announcement(anns)
    assert best is not None
    assert best.title == "平安银行：2025年年度报告"


def test_pick_best_excludes_all_noise():
    anns = [
        _ann("平安银行：2025年年度报告摘要"),
        _ann("平安银行：2025年年度报告（英文版）"),
    ]
    assert pick_best_announcement(anns) is None


def test_pick_best_latest_wins_when_same_score():
    anns = [
        _ann("甲公司：2025年年度报告", "2026-03-20"),
        _ann("甲公司：2025年年度报告", "2026-04-30"),
    ]
    best = pick_best_announcement(anns)
    assert best is not None
    assert best.announcement_time == "2026-04-30"


def test_parse_period_valid():
    assert ReportFetcher._parse_period("2025-年报") == ("2025", "年报")
    assert ReportFetcher._parse_period("2025-一季报") == ("2025", "一季报")


def test_parse_period_invalid():
    for bad in ["2025", "2025-月报", "abc-年报"]:
        try:
            ReportFetcher._parse_period(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝非法 period: {bad}")
