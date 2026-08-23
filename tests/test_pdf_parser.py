"""pdf_parser 纯逻辑单元测试: 标题识别 / 报表定位 / 章节切块。"""

from finance_agent.parsing.pdf_parser import (
    PageText,
    consolidated_statement_name,
    is_section_title,
    locate_statements,
    split_into_chunks,
    statement_text_from_pages,
)


# ---------- 报表区域保护: 编号行项目不能切碎报表 ----------


def test_numbered_items_inside_statement_do_not_split():
    # 真实 bug 复现: 利润表内部的 "一、营业总收入" 是行项目, 不是章节边界
    pages = [
        PageText(page=1, text="合并利润表\n一、营业总收入 5,000\n二、营业总成本 3,000"),
        PageText(page=2, text="三、营业利润 2,000\n四、利润总额 1,900"),
    ]
    chunks = split_into_chunks(pages)
    assert len(chunks) == 1                       # 报表保持完整, 未被切碎
    assert chunks[0].section == "合并利润表"
    assert chunks[0].page_end == 2


def test_note_headers_outside_statement_do_split():
    # 附注区的 "五、重要会计政策及会计估计" 是真边界
    pages = [
        PageText(page=1, text="财务报告\n审计意见正文"),
        PageText(page=2, text="三、公司基本情况\n公司成立于..."),
        PageText(page=3, text="五、重要会计政策及会计估计\n会计政策内容"),
    ]
    chunks = split_into_chunks(pages)
    sections = [c.section for c in chunks]
    assert "三、公司基本情况" in sections
    assert "五、重要会计政策及会计估计" in sections


# ---------- 标题识别 ----------


def test_consolidated_title_exact():
    assert consolidated_statement_name("合并资产负债表") == "合并资产负债表"
    assert consolidated_statement_name("合并利润表") == "合并利润表"
    assert consolidated_statement_name("合并现金流量表") == "合并现金流量表"


def test_consolidated_title_with_suffix():
    # 带括号后缀的标题也能识别
    assert consolidated_statement_name("合并资产负债表（未经审计）") == "合并资产负债表"


def test_consolidated_title_rejects_noise():
    # 目录罗列行: 前缀带序号
    assert consolidated_statement_name("1、合并及公司资产负债表（未经审计）") is None
    # 母公司单体报表不是合并报表
    assert consolidated_statement_name("母公司资产负债表") is None
    # 正文长句不以标题形式出现
    assert consolidated_statement_name("我们编制了合并资产负债表以及相关附注") is None
    # 超长行拒绝
    assert consolidated_statement_name("合并资产负债表" + "的" * 30) is None


def test_is_section_title():
    assert is_section_title("第一节 释义")
    assert is_section_title("第二节 公司简介和主要财务指标")
    assert is_section_title("财务报告")
    assert is_section_title("母公司利润表")
    assert not is_section_title("公司简介和主要财务指标详见后文第十节的详细说明内容")


# ---------- 报表定位(构造页数据, 不碰文件) ----------


def _make_pages() -> list[PageText]:
    return [
        PageText(page=1, text="平安银行股份有限公司 2026 年第一季度报告\n重要提示"),
        PageText(page=2, text="1、合并及公司资产负债表（未经审计）\n2、合并及公司利润表（未经审计）"),
        PageText(page=3, text="合并资产负债表\n货币资金 100,000"),
        PageText(page=4, text="发放贷款和垫款 200,000\n资产总计 300,000"),
        PageText(page=5, text="合并利润表\n营业收入 50,000"),
        PageText(page=6, text="营业支出 30,000"),
        PageText(page=7, text="合并现金流量表\n经营活动现金流量净额 10,000"),
    ]


def test_locate_statements_ranges():
    locs = locate_statements(_make_pages())
    assert locs["合并资产负债表"] == (3, 4)   # 到下一标题前一页
    assert locs["合并利润表"] == (5, 6)
    assert locs["合并现金流量表"] == (7, 7)   # 最后一个标题: 到文档末页


def test_locate_statements_same_page_boundary():
    # 两个标题在同一页: 前一个报表终点是本页
    pages = [
        PageText(page=1, text="合并资产负债表\n资产 1\n合并利润表\n收入 2"),
        PageText(page=2, text="净利润 3"),
    ]
    locs = locate_statements(pages)
    assert locs["合并资产负债表"] == (1, 1)
    assert locs["合并利润表"] == (1, 2)


# ---------- 章节切块 ----------


def test_split_into_chunks_by_section():
    chunks = split_into_chunks(_make_pages())
    sections = [c.section for c in chunks]
    assert "合并资产负债表" in sections
    assert "合并利润表" in sections
    bs = next(c for c in chunks if c.section == "合并资产负债表")
    assert bs.page == 3 and bs.page_end == 4
    assert "货币资金" in bs.text


def test_split_into_chunks_long_section():
    # 长章节按页再切分(切块粒度最小是页, 单页超长则保持整页)
    long_pages = [
        PageText(page=1, text="前言内容"),
        PageText(page=2, text="财务报告\n" + "附注" * 30),      # ~65 字符
        PageText(page=3, text="附注续" * 30),                   # 90 字符
        PageText(page=4, text="附注续" * 28),                   # 84 字符
    ]
    chunks = split_into_chunks(long_pages, max_chars=100)
    fin = [c for c in chunks if c.section == "财务报告"]
    assert len(fin) >= 2                       # 被按页切开
    assert all(len(c.text) <= 100 for c in fin)
    assert fin[0].page == 2                    # 页码范围保持正确
    assert fin[-1].page_end == 4


# ---------- 报表尾部与下一张报表标题同页(真实 bug 复现) ----------


def test_statement_tail_on_same_page_as_next_title():
    # 茅台 2025 年报真实场景: 合并现金流量表的筹资部分与
    # "母公司现金流量表"标题同在一页, 不能整页丢弃
    pages = [
        PageText(page=1, text="合并现金流量表\n一、经营活动净额 100"),
        PageText(page=2, text="三、筹资活动净额 50\n母公司现金流量表\n一、经营活动净额 30"),
    ]
    text = statement_text_from_pages(pages, "合并现金流量表")
    assert "筹资活动净额 50" in text          # 同页尾部保留
    assert "母公司现金流量表" not in text      # 下一张报表标题截断
    assert "经营活动净额 30" not in text       # 母公司数据不混入


def test_statement_tail_truncation_next_page_boundary():
    # 下一边界在独立页: 行为与旧逻辑一致(整页保留)
    pages = [
        PageText(page=1, text="合并资产负债表\n资产总计 100"),
        PageText(page=2, text="负债合计 60"),
        PageText(page=3, text="母公司资产负债表\n资产总计 80"),
    ]
    text = statement_text_from_pages(pages, "合并资产负债表")
    assert "资产总计 100" in text
    assert "负债合计 60" in text
    assert "母公司" not in text


def test_statement_text_not_found_raises():
    import pytest

    pages = [PageText(page=1, text="目录")]
    with pytest.raises(ValueError, match="未找到"):
        statement_text_from_pages(pages, "合并利润表")
