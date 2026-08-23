# FinSight — A股财报分析智能体

输入股票代码，自动下载财报 PDF，完成结构化抽取、财务比率分析、同比异常预警，生成**每条结论可溯源到原文页码**的分析报告。

> ⚠️ 本项目仅用于学习研究，输出不构成投资建议。所有数据来自公开渠道（巨潮资讯、AKShare）。

## 快速开始

```bash
# 1. 安装(建议 Python 3.11+)
pip install -e ".[dev]"        # 或 uv sync

# 2. 配置
cp .env.example .env           # 填入 LLM_API_KEY

# 3. 跑测试(规则模块无需 API key)
pytest

# 4. 启动 API(第9周后可用)
uvicorn finance_agent.api.app:app --reload
```

## 架构

```
财报 PDF (巨潮资讯)
   │
   ▼
Data 层      下载缓存 + AKShare 交叉核对
Parsing 层   PyMuPDF 章节切块(保留页码) → LLM+Pydantic 结构化抽取 → 会计恒等式校验
Analysis 层  比率计算 + 同比异常检测(纯规则, 可测)
Agent 层     LangGraph: parse → extract → validate → analyze → debate → report
RAG 层       Chroma 章节级向量库, 溯源粒度到页码
服务/评测层  FastAPI · Langfuse 追踪 · 字段级评测集
```

详细设计见 [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)（含 12 周里程碑与评测集设计）。

## 评测

```bash
python -m finance_agent.eval.run_eval
```

指标：字段级 precision / recall（数值容忍 0.5% 相对误差），分行业统计。

## License

MIT
