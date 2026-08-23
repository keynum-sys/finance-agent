"""财报 PDF 下载与缓存（巨潮资讯网 www.cninfo.com.cn）。

设计说明（面试可讲）:
1. 巨潮没有官方公开 API，这里用的是其网页版搜索接口（POST 表单），
   属于"逆向网页接口"——稳定但可能变更，所以做了结构化异常 + 可替换。
2. 搜索结果往往包含「摘要」「英文版」「已取消」等干扰项，
   用纯函数 pick_best_announcement() 做筛选，方便单元测试。
3. 下载结果缓存到 data_cache/reports/{code}/{period}.pdf，
   评测时避免重复下载同一份文件（评测集要跑很多遍）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from finance_agent.config import settings

CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_TOPSEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_PDF_BASE = "http://static.cninfo.com.cn"  # adjunctUrl 已含 finalpage/ 前缀

# 报告类型 -> 巨潮公告分类码（category 参数）
CATEGORY_MAP = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
}

# 标题里出现这些词的公告要排除
EXCLUDE_KEYWORDS = ("摘要", "英文版", "已取消", "更正", "补充", "取消", "说明书")


@dataclass
class Announcement:
    """一条公告搜索结果。"""
    title: str
    pdf_url: str
    announcement_time: str  # 格式 yyyy-mm-dd


def pick_best_announcement(announcements: list[Announcement]) -> Announcement | None:
    """从搜索结果中挑出真正的财报正文。

    筛选规则（按优先级）:
    1. 排除摘要/英文版/更正等干扰项
    2. 标题以「年度报告」「季度报告」「半年度报告」结尾的优先
    3. 同分情况下取时间最新的
    """
    candidates: list[Announcement] = []
    for a in announcements:
        if any(kw in a.title for kw in EXCLUDE_KEYWORDS):
            continue
        candidates.append(a)
    if not candidates:
        return None

    def score(a: Announcement) -> tuple[int, str]:
        # 标题干净（以"报告"结尾且不带括号补充说明）得分更高
        clean_title = a.title.endswith("报告") and "(" not in a.title and "（" not in a.title
        return (1 if clean_title else 0, a.announcement_time)

    return max(candidates, key=score)


class ReportFetcher:
    """按 股票代码 + 报告期 下载财报 PDF，带本地缓存。"""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.cache_dir = settings.data_dir / "reports"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 巨潮接口对 UA 有校验，带上浏览器 UA
        self._client = client or httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=30.0,
        )
        self._org_id_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def download(self, code: str, period: str, force: bool = False) -> Path:
        """下载指定股票、指定报告期(如 '2025-年报')的财报 PDF，返回本地路径。

        已有缓存时直接返回（除非 force=True）。
        """
        cache_path = self.cache_dir / code / f"{period}.pdf"
        if cache_path.exists() and not force:
            return cache_path

        year, report_type = self._parse_period(period)
        announcement = self._search_one(code, year, report_type)
        if announcement is None:
            raise FileNotFoundError(f"巨潮未找到 {code} {period} 的财报正文公告")

        pdf_bytes = self._download_pdf(announcement.pdf_url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pdf_bytes)
        return cache_path

    def list_cached(self, code: str) -> list[Path]:
        """列出某股票已缓存的所有财报 PDF。"""
        return sorted((self.cache_dir / code).glob("*.pdf"))

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_period(period: str) -> tuple[str, str]:
        """'2025-年报' -> ('2025', '年报')。"""
        year, _, report_type = period.partition("-")
        if not (len(year) == 4 and year.isdigit()) or report_type not in CATEGORY_MAP:
            raise ValueError(
                f"period 格式应为 'YYYY-年报|半年报|一季报|三季报'，收到: {period!r}"
            )
        return year, report_type

    def _search_one(self, code: str, year: str, report_type: str) -> Announcement | None:
        """搜索并返回最佳的一条公告。"""
        announcements = self.search_announcements(code, year, report_type)
        return pick_best_announcement(announcements)

    def search_announcements(
        self, code: str, year: str, report_type: str
    ) -> list[Announcement]:
        """调用巨潮搜索接口，返回该股票该报告期的公告列表。"""
        org_id = self._get_org_id(code)
        se_date = f"{year}-01-01~{int(year) + 1}-06-30"  # 年报可能次年才发布，留足窗口
        resp = self._client.post(
            CNINFO_SEARCH_URL,
            data={
                "pageNum": 1,
                "pageSize": 30,
                "column": "szse",
                "tabName": "fulltext",
                "stock": f"{code},{org_id}",
                "category": CATEGORY_MAP[report_type],
                "seDate": se_date,
                "isHLtitle": "true",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        results: list[Announcement] = []
        for item in payload.get("announcements") or []:
            adjunct_url = item.get("adjunctUrl", "")
            if not adjunct_url:
                continue
            results.append(
                Announcement(
                    title=item.get("announcementTitle", "").replace("<em>", "").replace("</em>", ""),
                    pdf_url=f"{CNINFO_PDF_BASE}/{adjunct_url}",
                    announcement_time=time.strftime(
                        "%Y-%m-%d", time.localtime(item["announcementTime"] / 1000)
                    ),
                )
            )
        return results

    def _get_org_id(self, code: str) -> str:
        """巨潮搜索接口需要 orgId（公司内部 ID），先通过 topSearch 查询。结果缓存。"""
        if code in self._org_id_cache:
            return self._org_id_cache[code]
        resp = self._client.post(
            CNINFO_TOPSEARCH_URL,
            data={"keyWord": code, "maxNum": 10},
        )
        resp.raise_for_status()
        for item in resp.json():
            if item.get("code") == code:
                org_id = item["orgId"]
                self._org_id_cache[code] = org_id
                return org_id
        raise ValueError(f"巨潮查不到股票代码 {code} 对应的公司")

    def _download_pdf(self, url: str) -> bytes:
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.content
