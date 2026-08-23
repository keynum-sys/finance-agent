"""PDF 解析:按章节切块,每块保留页码元数据(溯源的基础)。

第 3 周实现。设计要点:
- 用 PyMuPDF 提取文本与表格,记录每个 chunk 的 (page_start, page_end)
- 按财报标准章节切分:重要提示/资产负债表/利润表/现金流量表/管理层讨论
- chunk 同时是 RAG 向量库的入库单元——溯源粒度即由此决定
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf


@dataclass
class Chunk:
    page: int              # 起始页码(1-based, 溯引用)
    section: str           # 章节名, 如 "合并资产负债表"
    text: str


def parse_report(pdf_path: str) -> list[Chunk]:
    """解析财报 PDF,返回带页码的章节级 chunk 列表。

    TODO: PyMuPDF 逐页提取 -> 章节标题识别 -> 切块。
    """
    doc = pymupdf.open(pdf_path)
    chunks: list[Chunk] = []
    # TODO: 实现章节识别与切块
    doc.close()
    return chunks
