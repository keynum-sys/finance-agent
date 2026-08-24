# -*- coding: utf-8 -*-
"""从东方财富(datacenter)拉取全量 A 股清单(代码+名称), 写入 data/a_shares.json。

覆盖 沪市主板/科创板/深市主板/创业板(共约 5000 支)。名称做 NFKC 归一化(全角空格、
全角字母统一), 便于离线模糊匹配。运行: python data/build_company_list.py
"""
from __future__ import annotations

import json
import unicodedata
import urllib.request
from pathlib import Path

REPORT = "RPT_DMSK_TS_STOCKNEW"
PAGE_SIZE = 500  # 服务端单页上限, 需分页
HERE = Path(__file__).resolve().parent
OUT = HERE / "a_shares.json"

VALID_PREFIXES = ("60", "68", "00", "30")  # 沪市/科创/深市/创业; 排除 B股(20/90)与北交所


def _norm_name(name: str) -> str:
    # 全角->半角, 去空格
    return unicodedata.normalize("NFKC", name).replace(" ", "").strip()


def fetch() -> dict[str, str]:
    stocks: dict[str, str] = {}
    pn = 1
    while True:
        url = (
            f"https://datacenter-web.eastmoney.com/api/data/v1/get?reportName={REPORT}"
            f"&columns=SECURITY_CODE,SECURITY_NAME_ABBR&pageSize={PAGE_SIZE}"
            f"&pageNumber={pn}&sortColumns=SECURITY_CODE&sortTypes=1"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://datacenter-web.eastmoney.com/"},
        )
        r = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
        data = (r.get("result") or {}).get("data") or []
        if not data:
            break
        for it in data:
            code = (it.get("SECURITY_CODE") or "").strip()
            name = (it.get("SECURITY_NAME_ABBR") or "").strip()
            if not code or not name:
                continue
            if not code.isdigit() or len(code) != 6:
                continue
            if code[:2] not in VALID_PREFIXES:
                continue
            stocks[code] = _norm_name(name)
        if len(data) < PAGE_SIZE:
            break
        pn += 1
    return stocks


def main() -> None:
    stocks = fetch()
    out = [{"code": c, "name": n} for c, n in sorted(stocks.items())]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"写入 {len(out)} 支 A 股到 {OUT}")


if __name__ == "__main__":
    main()
