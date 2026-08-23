# -*- coding: utf-8 -*-
"""验证 DeepSeek API 连通性：发送一条消息，确认 key 有效、模型正常响应。

用法:
    python examples/test_llm.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.config import settings


def main() -> None:
    print("=" * 50)
    print("DeepSeek API 连通性测试")
    print("=" * 50)
    print(f"Base URL: {settings.llm_base_url}")
    print(f"Model:    {settings.llm_model}")
    print(f"API Key:  {settings.llm_api_key[:8]}****")
    print()

    payload = json.dumps({
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "你是一个金融助手, 回复尽量简短。"},
            {"role": "user", "content": "用一句话解释什么是 ROE。"},
        ],
        "max_tokens": 100,
        "temperature": 0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url=f"{settings.llm_base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print("[成功] 模型回复:")
        print(f"    {content.strip()}")
        print()
        print(f"Token 用量: 输入 {usage.get('prompt_tokens', '?')} + "
              f"输出 {usage.get('completion_tokens', '?')} = "
              f"总计 {usage.get('total_tokens', '?')}")
        print()
        print("结论: API key 有效, DeepSeek 连通正常。")
        print("第 4 周实现 LLM 结构化抽取时可直接使用。")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[失败] HTTP {e.code}: {body[:200]}")
        if e.code == 401:
            print("API key 无效, 检查 .env 中的 LLM_API_KEY 是否正确")
        sys.exit(1)
    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
