"""PDF 解析: 逐页提取文本 -> 定位三大合并报表 -> 切成带页码的章节 chunk。

设计要点:
- 溯源的最小单位是 chunk, 每个 chunk 记录 (section, page_start, page_end)
- 三大报表定位只认"合并报表"(母公司单体报表排除)——业务上分析主体是集团
- 标题识别/切块逻辑全部写成纯函数, 不碰文件, 便于单元测试
- chunk 同时是后续 RAG 向量库的入库单元, 超长章节按页再切分
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

# --------------------------------------------------------------------------
# 标题识别(纯函数, 规则来自对真实财报的观察)
# --------------------------------------------------------------------------

# 合并三大报表标题: "合并资产负债表" / "合并利润表" / "合并现金流量表",
# 允许带括号后缀如 "(未经审计)"。目录行的前缀是 "1、" 之类, 不会误匹配。
_CONSOLIDATED_RE = re.compile(
    r"^合并(资产负债表|利润表|现金流量表)(?:[（(][^（）()]*[）)])?\s*$"
)

# 母公司报表标题: 用于章节边界, 但不算"三大报表定位"结果
_PARENT_RE = re.compile(
    r"^母公司(资产负债表|利润表|现金流量表|所有者权益变动表)(?:[（(][^（）()]*[）)])?\s*$"
)

# 章节标题(强边界): 年报的 "第X节 XXX" + 常见章节名 + 各类报表标题
_SECTION_RE = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十]+节\s*.+"  # 第一节 释义 / 第二节 公司简介...
    r"|重要提示"
    r"|目录"
    r"|释义"
    r"|管理层讨论与分析"
    r"|财务报告"
    r"|合并(?:资产负债表|利润表|现金流量表|所有者权益变动表).*"
    r"|母公司(?:资产负债表|利润表|现金流量表|所有者权益变动表).*"
    r")$"
)

# 附注编号标题(弱边界): "五、重要会计政策及会计估计" / "七、合并财务报表项目注释"。
# 注意: 报表内部的行项目也有编号行("一、营业总收入" "一、经营活动产生的现金流量"
# "二、本年期初余额"等), 它们几乎都包含财务科目关键词——用黑名单排除。
_NOTE_HEADER_RE = re.compile(r"^[一二三四五六七八九十]+、.{2,25}$")

# 报表行项目黑名单: 命中即认为该编号行是报表内部科目, 不算章节边界
_STATEMENT_ITEM_KW = re.compile(
    r"余额|变动|营业|利润|收益|现金|活动|资产|负债|权益"
)

# 任意一张报表的标题(合并或母公司), 用于判断"当前处于报表区域内"
_STATEMENT_TITLE_RE = re.compile(
    r"^(?:合并|母公司)"
    r"(?:资产负债表|利润表|现金流量表|所有者权益变动表)"
    r"(?:[（(][^（）()]*[）)])?\s*$"
)

# 标题行长度上限: 防止把恰好以这些词开头的正文长句误认成标题
_MAX_TITLE_LEN = 25


def consolidated_statement_name(line: str) -> str | None:
    """若该行是合并报表标题, 返回报表名(如 '合并资产负债表'), 否则 None。"""
    line = line.strip()
    if len(line) > _MAX_TITLE_LEN:
        return None
    m = _CONSOLIDATED_RE.match(line)
    return "合并" + m.group(1) if m else None


def is_section_title(line: str) -> bool:
    """判断该行是否为章节标题(强边界)。"""
    line = line.strip()
    return len(line) <= _MAX_TITLE_LEN and bool(_SECTION_RE.match(line))


def is_note_header(line: str) -> bool:
    """判断该行是否为附注编号标题(弱边界)。

    排除报表内部行项目: "一、营业总收入" 是利润表科目, 不是章节标题。
    """
    line = line.strip()
    if len(line) > _MAX_TITLE_LEN or not _NOTE_HEADER_RE.match(line):
        return False
    return not _STATEMENT_ITEM_KW.search(line)


def is_statement_title(line: str) -> bool:
    """判断当前章节名是否为报表标题(用于报表区域保护)。"""
    line = line.strip()
    return len(line) <= _MAX_TITLE_LEN and bool(_STATEMENT_TITLE_RE.match(line))


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass
class PageText:
    """单页文本, page 为 1-based 页码(溯源用)。"""

    page: int
    text: str

    @property
    def lines(self) -> list[str]:
        return [ln.strip() for ln in self.text.split("\n") if ln.strip()]


@dataclass
class Chunk:
    """章节级文本块: RAG 入库单元, 溯源粒度由它决定。"""

    page: int        # 起始页码(1-based)
    page_end: int    # 结束页码(含)
    section: str     # 章节名, 如 "合并资产负债表"
    text: str


# --------------------------------------------------------------------------
# 提取与定位
# --------------------------------------------------------------------------


def extract_pages(pdf_path: str) -> list[PageText]:
    """逐页提取文本。唯一的 IO 函数, 其余全是纯函数。"""
    doc = pymupdf.open(pdf_path)
    try:
        return [PageText(page=i, text=page.get_text()) for i, page in enumerate(doc, 1)]
    finally:
        doc.close()


def _collect_boundaries(pages: list[PageText]) -> list[tuple[int, str]]:
    """收集全部章节边界: [(页码, 标题)], 按文档顺序。"""
    return [
        (p.page, ln) for p in pages for ln in p.lines if is_section_title(ln)
    ]


def locate_statements(pages: list[PageText]) -> dict[str, tuple[int, int]]:
    """定位三大合并报表的页码范围。

    返回如 {"合并资产负债表": (56, 58), "合并利润表": (61, 62), ...}。
    起点是标题所在页; 终点是下一个章节标题出现的前一页(同页则算同页)。

    注意: 终点是"页粒度"的保守值。报表尾部可能与下一张报表标题同页
    (见 statement_text_from_pages 的截断处理), 需要精确文本时不要直接
    拿页码范围拼接。
    """
    boundaries = _collect_boundaries(pages)
    # 找到三张合并报表标题的位置
    result: dict[str, tuple[int, int]] = {}
    last_page = pages[-1].page if pages else 0
    for idx, (page, title) in enumerate(boundaries):
        name = consolidated_statement_name(title)
        if name is None or name in result:
            continue
        # 终点: 下一个边界的前一页; 同页有下一边界则终点就是本页
        if idx + 1 < len(boundaries):
            next_page = boundaries[idx + 1][0]
            end = page if next_page == page else next_page - 1
        else:
            end = last_page
        result[name] = (page, end)
    return result


def statement_text_from_pages(pages: list[PageText], statement: str) -> str:
    """抽取指定合并报表的完整文本(第 4-5 周 LLM 抽取的输入)。

    关键细节: 报表尾部常与下一张报表标题同页(如茅台 2025 年报中,
    合并现金流量表的筹资部分与"母公司现金流量表"标题同在一页)。
    因此尾页文本要在下一个标题"行"处截断, 而不是整页丢弃。
    """
    boundaries = _collect_boundaries(pages)
    idx = next(
        (
            i
            for i, (_pg, title) in enumerate(boundaries)
            if consolidated_statement_name(title) == statement
        ),
        None,
    )
    if idx is None:
        raise ValueError(f"未找到 {statement}, 已定位: {list(locate_statements(pages))}")

    start = boundaries[idx][0]
    if idx + 1 < len(boundaries):
        end, next_title = boundaries[idx + 1]
    else:
        end, next_title = pages[-1].page, None

    parts: list[str] = []
    for p in pages:
        if p.page < start or p.page > end:
            continue
        if next_title is not None and p.page == end:
            # 尾页: 在下一张报表标题行处截断
            lines = p.text.split("\n")
            cut = next(
                (i for i, ln in enumerate(lines) if ln.strip() == next_title),
                None,
            )
            parts.append("\n".join(lines[:cut]) if cut is not None else p.text)
        else:
            parts.append(p.text)
    return "\n".join(parts)


def extract_statement_text(pdf_path: str, statement: str) -> str:
    """抽取指定合并报表的完整文本(文件版入口)。"""
    return statement_text_from_pages(extract_pages(pdf_path), statement)


# --------------------------------------------------------------------------
# 章节切块
# --------------------------------------------------------------------------


def split_into_chunks(pages: list[PageText], max_chars: int = 6000) -> list[Chunk]:
    """把整份财报切成章节级 chunk; 超长章节按页再切分。

    页与页之间插入 "\\n\\f" 换页符, 供下游感知页边界。
    """
    chunks: list[Chunk] = []
    current_section = "前言"
    current_pages: list[PageText] = []

    def flush() -> None:
        if not current_pages:
            return
        text = "\n\f\n".join(p.text for p in current_pages)
        # 超长章节: 按页再切, 保证每个 chunk 不超过 max_chars 量级
        if len(text) > max_chars and len(current_pages) > 1:
            _flush_by_page(current_section, current_pages)
        else:
            chunks.append(
                Chunk(
                    page=current_pages[0].page,
                    page_end=current_pages[-1].page,
                    section=current_section,
                    text=text,
                )
            )

    def _flush_by_page(section: str, pgs: list[PageText]) -> None:
        buf: list[PageText] = []
        size = 0
        for p in pgs:
            if buf and size + len(p.text) > max_chars:
                chunks.append(
                    Chunk(
                        page=buf[0].page,
                        page_end=buf[-1].page,
                        section=section,
                        text="\n\f\n".join(b.text for b in buf),
                    )
                )
                buf, size = [], 0
            buf.append(p)
            size += len(p.text)
        if buf:
            chunks.append(
                Chunk(
                    page=buf[0].page,
                    page_end=buf[-1].page,
                    section=section,
                    text="\n\f\n".join(b.text for b in buf),
                )
            )

    for p in pages:
        # 边界识别分两级:
        # 1. 强边界(报表/章节标题)总是生效
        # 2. 弱边界(附注编号标题)已用关键词黑名单排除报表内部行项目
        title = next((ln for ln in p.lines if is_section_title(ln)), None)
        if title is None:
            title = next((ln for ln in p.lines if is_note_header(ln)), None)
        if title is not None:
            flush()
            current_pages.clear()
            current_section = title
        current_pages.append(p)
    flush()
    return chunks


# 对外主入口 ---------------------------------------------------------------


def parse_report(pdf_path: str) -> list[Chunk]:
    """解析财报 PDF, 返回带页码的章节级 chunk 列表。"""
    return split_into_chunks(extract_pages(pdf_path))
