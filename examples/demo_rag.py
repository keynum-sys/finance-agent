"""RAG 附注问答端到端演示: 茅台年报入库 -> 向量检索 -> DeepSeek 带页码溯源回答。

用法:
    python examples/demo_rag.py [code] [period] [问题]
默认: 600519 2025-年报, 附注示例问题。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finance_agent.data.fetcher import ReportFetcher
from finance_agent.parsing.pdf_parser import extract_pages, split_into_chunks
from finance_agent.rag.store import ReportVectorStore

DEFAULT_QUESTIONS = [
    "应收账款的坏账准备是怎么计提的?",
    "存货主要由什么构成?",
    "公司的分红政策是什么?",
]


def main() -> None:
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    period = sys.argv[2] if len(sys.argv) > 2 else "2025-年报"
    questions = [sys.argv[3]] if len(sys.argv) > 3 else DEFAULT_QUESTIONS

    print(f"== 1. 获取报告: {code} {period}")
    pdf = ReportFetcher().download(code, period)
    print(f"   {pdf}")

    print("== 2. 解析并切分章节")
    chunks = split_into_chunks(extract_pages(str(pdf)))
    print(f"   {len(chunks)} 个章节 chunk")

    print("== 3. 子分块 + embedding 入库 (首次运行需加载 bge-small-zh 模型)")
    store = ReportVectorStore()
    n = store.add_chunks(code, period, chunks)
    print(f"   入库 {n} 个子块")

    for q in questions:
        print(f"\n{'=' * 60}\n问题: {q}")
        answer, citations = store.query_with_citations(q, code, period=period, top_k=4)
        print(f"\n回答:\n{answer}")
        print("\n引用:")
        for c in citations:
            print(f"  [第{c.page}页 · {c.section}] {c.snippet[:60]}...")


if __name__ == "__main__":
    main()
