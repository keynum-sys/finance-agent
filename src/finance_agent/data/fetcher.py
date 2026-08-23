"""财报 PDF 下载与缓存。

第 3 周实现。核心思路:
1. 通过巨潮资讯搜索接口按股票代码 + 报告期查询 PDF 直链
2. 下载后缓存到 data_cache/{code}/{period}.pdf
3. 提供 list_cached_reports() 供后续模块使用
"""

from __future__ import annotations

from pathlib import Path

import httpx

from finance_agent.config import settings

CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"


class ReportFetcher:
    def __init__(self) -> None:
        self.cache_dir = settings.data_dir / "reports"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, code: str, period: str) -> Path:
        """下载指定股票、指定报告期(如 '2025-年报')的财报 PDF。

        TODO: 实现巨潮搜索 + 下载 + 缓存逻辑。
        """
        raise NotImplementedError

    def list_cached(self, code: str) -> list[Path]:
        """列出某股票已缓存的所有财报 PDF。"""
        return sorted((self.cache_dir / code).glob("*.pdf"))
