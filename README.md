# FinSight — A股财报分析智能体

输入股票代码，自动下载财报 PDF，完成结构化抽取、财务比率分析、多空辩论、附注问答，生成**每条结论可溯源到原文页码**的分析报告。

> ⚠️ 本项目仅用于学习研究，输出不构成投资建议。所有数据来自公开渠道（巨潮资讯、AKShare）。

## 快速开始

```bash
# 1. 安装(建议 Python 3.11+)
pip install -e ".[dev]"        # 或 uv sync

# 2. 配置
cp .env.example .env           # 填入 LLM_API_KEY
#    llm_api_key=sk-xxxx
#    llm_base_url=https://api.deepseek.com/v1
#    llm_model=deepseek-chat

# 3. 跑测试(87 个, 规则模块无需 API key, 全离线)
pytest

# 4. 启动 API
uvicorn finance_agent.api.app:app --reload
```

## 三种调用方式

### 1. Python API

```python
from finance_agent.agents.graph import run_pipeline

state = run_pipeline("600519", "2025-年报", question="坏账准备怎么计提?")
print(state["report_md"])        # 完整 Markdown 报告
print(state["qa"]["citations"])  # 附注问答的页码溯源
```

不传 `question` 时不启用 RAG（零开销）；传入则自动入库并追加附注问答章节。

### 2. HTTP API（FastAPI）

```bash
uvicorn finance_agent.api.app:app

# 完整流水线(同步, 约 1-3 分钟; 传 question 则含附注问答)
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "600519", "period": "2025-年报", "question": "分红政策是什么?"}'

# 纯附注问答(不跑抽取/辩论, 前提: 报告已入库过)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"code": "600519", "period": "2025-年报", "question": "存货由什么构成?"}'
```

### 3. Demo 脚本

```bash
python examples/demo_rag_graph.py "公司的分红政策是什么?"  # 一张图跑全流程
python examples/demo_debate.py                             # 仅多空辩论
python examples/demo_rag.py                                # 仅 RAG 检索问答
```

## 架构

```
财报 PDF (巨潮资讯)
   │
   ▼
Data 层      下载缓存 + AKShare 交叉核对
Parsing 层   PyMuPDF 章节切块(保留页码) → LLM+Pydantic 结构化抽取
             → 单位检测(百万元/亿元换算) → 会计恒等式校验
Analysis 层  比率计算 + 同比异常检测(纯规则, 可测)
Agent 层     LangGraph:
               download → extract → analyze ─┬→ debate ─┬(question?)→ rag_qa ─┐
                                              └→ index ──┴───────────────────┴→ report
RAG 层       bge-small-zh 子分块(页级) → Chroma 向量库, 溯源粒度到页码
服务/评测层  FastAPI · 字段级评测集
```

详细设计见 [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)（含 12 周里程碑与评测集设计）。

## 评测

```bash
python -m finance_agent.eval.run_eval
```

- 金标集 `eval/golden.jsonl`：茅台 2025 年报 + 平安银行 2026 一季报（白酒+银行两行业，人工核对自 PDF 原文）
- 指标：字段级 precision / recall / **F1**（数值容忍 0.5% 相对误差），总体 + 分行业
- 当前总体 F1 **95.45%**。评测曾抓出银行股"百万元→元"换算错 10 倍的真实 bug（恒等式校验无法发现的等比例错误）

## License

MIT
