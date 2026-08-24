# -*- coding: utf-8 -*-
"""FinSight 财报分析 Agent — 交互式一键启动器。

双击项目根目录下的「运行demo.bat」即可启动本脚本。
可以直接说一家公司的名字（或代码/别名），智能体会去巨潮下载它的财报并回答；
也可以输入「随机」让系统随机抽一家公司来分析。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from finance_agent.agents.graph import run_pipeline
from finance_agent.rag.store import ReportVectorStore

# 预设题库: 覆盖分红/现金流/负债/营收/研发等多维度, 随机抽题时选用
QUESTION_BANK = [
    "公司的现金分红政策是什么？本期分红预案如何？",
    "应收账款的坏账准备是怎么计提的？账龄分布如何？",
    "本期经营活动现金流净额是多少？与净利润是否匹配？",
    "公司的资产负债率和有息负债情况如何？",
    "营业收入同比/环比增长情况如何？主要驱动因素是什么？",
    "研发投入占营业收入的比例是多少？同比变化如何？",
    "主营业务毛利率是多少？相比上年有何变化？",
    "是否存在大额商誉？商誉减值风险如何？",
    "前五大客户/供应商的集中度如何？",
    "关联交易的主要内容和规模是什么？",
    "员工总数和人均薪酬变化情况如何？",
    "货币资金是否存在受限情形？受限金额多少？",
]

# 内置常见 A 股公司库(名称, 代码, 别名)。按需可继续扩充。
# 名称/别名解析是离线匹配；未收录的公司可直接输入 6 位代码让智能体去抓。
COMPANY_DB = [
    ("贵州茅台", "600519", ["茅台"]),
    ("平安银行", "000001", []),
    ("宁德时代", "300750", ["宁德"]),
    ("中国平安", "601318", []),
    ("招商银行", "600036", ["招行"]),
    ("五粮液", "000858", []),
    ("美的集团", "000333", ["美的"]),
    ("格力电器", "000651", ["格力"]),
    ("比亚迪", "002594", []),
    ("海康威视", "002415", ["海康"]),
    ("隆基绿能", "601012", ["隆基"]),
    ("中国中免", "601888", ["中免"]),
    ("伊利股份", "600887", ["伊利"]),
    ("兴业银行", "601166", ["兴业"]),
    ("紫金矿业", "601899", ["紫金"]),
    ("长江电力", "600900", ["长电"]),
    ("恒瑞医药", "600276", ["恒瑞"]),
    ("三一重工", "600031", ["三一"]),
    ("工商银行", "601398", ["工行"]),
    ("农业银行", "601288", ["农行"]),
    ("中国神华", "601088", ["神华"]),
    ("海尔智家", "600690", ["海尔"]),
    ("山西汾酒", "600809", ["汾酒"]),
    ("海天味业", "603288", ["海天"]),
    ("药明康德", "603259", ["药明"]),
    ("中信证券", "600030", ["中信"]),
    ("泸州老窖", "000568", ["泸州"]),
    ("洋河股份", "002304", ["洋河"]),
    ("顺丰控股", "002352", ["顺丰"]),
    ("京东方A", "000725", ["京东方"]),
    ("立讯精密", "002475", ["立讯"]),
    ("东方财富", "300059", ["东财"]),
    ("迈瑞医疗", "300760", ["迈瑞"]),
    ("汇川技术", "300124", ["汇川"]),
    ("爱尔眼科", "300015", ["爱尔"]),
    ("潍柴动力", "000338", ["潍柴"]),
    ("中国石油", "601857", ["中石油"]),
    ("中芯国际", "688981", ["中芯"]),
    ("金山办公", "688111", ["金山"]),
    ("韦尔股份", "603501", ["韦尔"]),
    ("温氏股份", "300498", ["温氏"]),
    ("三花智控", "002050", ["三花"]),
    ("交通银行", "601328", ["交行"]),
    ("浦发银行", "600000", []),
    ("民生银行", "600016", []),
    ("中信银行", "601998", []),
    ("建设银行", "601939", ["建行"]),
    ("中国银行", "601988", ["中行"]),
    ("邮储银行", "601658", []),
    ("宁波银行", "002142", ["宁波"]),
    ("国泰君安", "601211", ["国君"]),
    ("华泰证券", "601688", ["华泰"]),
    ("中国银河", "601881", ["银河"]),
    ("广发证券", "000776", ["广发"]),
    ("东方证券", "600958", []),
    ("古井贡酒", "000596", []),
    ("今世缘", "603369", []),
    ("青岛啤酒", "600600", ["青啤"]),
    ("重庆啤酒", "600132", []),
    ("双汇发展", "000895", ["双汇"]),
    ("安井食品", "603345", ["安井"]),
    ("东鹏饮料", "605499", ["东鹏"]),
    ("涪陵榨菜", "002507", []),
    ("长城汽车", "601633", ["长城"]),
    ("上汽集团", "600104", ["上汽"]),
    ("广汽集团", "601238", ["广汽"]),
    ("长安汽车", "000625", ["长安"]),
    ("赛力斯", "601127", ["问界"]),
    ("福耀玻璃", "600660", ["福耀"]),
    ("亿纬锂能", "300014", ["亿纬"]),
    ("天齐锂业", "002466", ["天齐"]),
    ("赣锋锂业", "002460", ["赣锋"]),
    ("华友钴业", "603799", ["华友"]),
    ("阳光电源", "300274", ["阳光"]),
    ("通威股份", "600438", ["通威"]),
    ("晶澳科技", "002459", ["晶澳"]),
    ("天合光能", "688599", ["天合"]),
    ("北方华创", "002371", ["北方"]),
    ("中微公司", "688012", ["中微"]),
    ("兆易创新", "603986", ["兆易"]),
    ("澜起科技", "688008", ["澜起"]),
    ("长电科技", "600584", []),
    ("中兴通讯", "000063", ["中兴"]),
    ("科大讯飞", "002230", ["讯飞"]),
    ("寒武纪", "688256", ["寒武"]),
    ("智飞生物", "300122", ["智飞"]),
    ("云南白药", "000538", ["云白药"]),
    ("复星医药", "600196", ["复星"]),
    ("片仔癀", "600436", ["片仔癀"]),
    ("长春高新", "000661", ["长高"]),
    ("泰格医药", "300347", ["泰格"]),
    ("万科A", "000002", ["万科"]),
    ("保利发展", "600048", ["保利"]),
    ("中国建筑", "601668", ["中建"]),
    ("海螺水泥", "600585", ["海螺"]),
    ("东方雨虹", "002271", ["雨虹"]),
    ("中国石化", "600028", ["中石化"]),
    ("三峡能源", "600905", ["三峡"]),
    ("万华化学", "600309", ["万华"]),
    ("荣盛石化", "002493", []),
]

RANDOM_KEYWORDS = {"随机", "随机公司", "随便", "random", "r"}

# 连续追问循环中的退出 / 换公司 关键字
EXIT_KW = {"退出", "exit", "quit", "q", "结束"}
SWITCH_KW = {"换公司", "切换", "换一家", "下一家", "switch"}


def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def _match_company(raw: str):
    """名称/别名/代码 -> (name, code)；歧义返回 ('ambiguous', hits)；未匹配返回 None。"""
    q = _norm(raw)
    if q.isdigit() and len(q) == 6:
        for name, code, _ in COMPANY_DB:
            if code == q:
                return (name, code)
        return (q, q)  # 当作代码直用, 交给 fetcher 去巨潮抓
    for name, code, aliases in COMPANY_DB:
        if _norm(name) == q or q in (_norm(a) for a in aliases):
            return (name, code)
    hits = [(name, code) for name, code, _ in COMPANY_DB if q in _norm(name)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return ("ambiguous", hits)
    return None


def choose_company() -> tuple[str, str]:
    """交互选择公司: 支持名称/别名/6 位代码/随机。"""
    print(f"可分析公司示例(输入名称/别名/代码, 或'随机'): 共 {len(COMPANY_DB)} 家")
    print("  " + "、".join(n for n, _, _ in COMPANY_DB[:14]))
    while True:
        raw = input("请输入公司(名称/别名/代码，回车或'随机'=随机抽一家): ").strip()
        if not raw or _norm(raw) in RANDOM_KEYWORDS:
            name, code = random.choice(COMPANY_DB)[:2]
            print(f"[随机抽公司] {name} ({code})")
            return name, code
        res = _match_company(raw)
        if res is None:
            print("未收录该公司。可输入库中名称/别名、6 位代码，或输入'随机'。")
            continue
        if isinstance(res, tuple) and res[0] == "ambiguous":
            print("匹配到多家，请更精确输入(全称或代码):")
            for n, c in res[1]:
                print(f"  {n} ({c})")
            continue
        return res


def ask(prompt: str, default: str) -> str:
    val = input(prompt).strip()
    return val or default


def analyze_company(name: str, code: str, period: str) -> str:
    """分析一家公司并进入连续追问。返回 'exit'（结束程序）或 'switch'（换下一家）。"""
    raw = input("可选: 输入附注问题 (直接回车或输入'随机' = 随机抽一题): ").strip()
    if not raw or _norm(raw) in {"随机", "随机提问", "random", "r"}:
        question = random.choice(QUESTION_BANK)
        print(f"[随机抽题] {question}")
    else:
        question = raw

    print(f"\n开始分析 {name}({code}) {period}，首次会联网下载 PDF，请稍候...\n")
    store = ReportVectorStore()  # 复用同一向量库做后续连续追问
    try:
        state = run_pipeline(code, period, question=question, store=store, enable_rag=True)
    except Exception as e:  # noqa: BLE001  下载/解析失败给友好提示, 不闪退
        print(f"\n分析失败: {e}")
        print("提示: 可能是该公司/报告期暂无财报, 或网络问题。可换一家公司或报告期重试。")
        return "switch"

    print(f"\n入库子块数: {state.get('indexed_count', 'N/A')}")
    qa = state.get("qa")
    if qa:
        print(f"\n--- 答案 ---\n{qa['answer']}")
        print("\n--- 引用来源 ---")
        for c in qa["citations"]:
            print(f"  第{c['page']}页({c['section']}): {c['snippet'][:60]}...")

    # 连续追问: 复用已建好的向量索引, 不重新下载/分析/抽取
    ret: str = "exit"
    while True:
        q = input("\n继续追问（直接回车或输入'退出'结束；输入'换公司'可切换下一家）：").strip()
        if not q or _norm(q) in EXIT_KW:
            ret = "exit"
            break
        if _norm(q) in SWITCH_KW:
            ret = "switch"
            break
        try:
            ans, cits = store.query_with_citations(q, code, period)
        except Exception as e:
            print(f"追问失败: {e}")
            continue
        print(f"\n答: {ans}")
        if cits:
            print("引用: " + "；".join(f"第{c.page}页({c.section})" for c in cits))

    md = state["report_md"]
    print("\n--- 报告章节 ---")
    for line in md.splitlines():
        if line.startswith("#"):
            print(" ", line)

    safe = f"{code}_{period}".replace("/", "_")
    out = Path(__file__).resolve().parent / "output" / f"{safe}_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n完整报告已写入: {out}")
    return ret


def main() -> None:
    print("=" * 60)
    print("FinSight 财报分析 Agent — 一键运行")
    print("=" * 60)
    print("报告期格式: YYYY-年报 | YYYY-半年报 | YYYY-一季报 | YYYY-三季报")
    print("-" * 60)

    first = True
    while True:
        if not first:
            print("\n" + "=" * 60)
            print("下面换一家公司继续分析")
            print("=" * 60)
        first = False
        name, code = choose_company()
        period = ask("请输入报告期 (如 2025-年报，回车默认 2025-年报): ", "2025-年报")
        action = analyze_company(name, code, period)
        if action == "exit":
            break
        # action == "switch": 回到外层循环, 重新选择公司

    print("\n感谢使用 FinSight，再见。")


if __name__ == "__main__":
    main()
