#!/usr/bin/env python3
"""
测试脚本：诊断 OpenAI SDK 请求头问题
"""

import asyncio
import json
from pathlib import Path

import httpx

# ============================================================
# 配置文件路径
# ============================================================
CONFIG_FILE = Path("/root/emby-sign/.signer/.openai_config.json")


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as fp:
            return json.load(fp)
    raise FileNotFoundError(f"配置文件不存在: {CONFIG_FILE}")


async def test_headers(cfg: dict):
    """测试不同请求头组合"""

    api_url = f"{cfg.get('base_url')}/chat/completions"

    payload = {
        "model": cfg.get("model", "gpt-4o"),
        "temperature": 0.1,
        "messages": [
            {"role": "user", "content": "Hello, reply with OK."}
        ]
    }

    # OpenAI SDK 默认会添加的请求头
    header_tests = [
        {
            "name": "1. 最简请求头 (httpx默认)",
            "headers": {
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            }
        },
        {
            "name": "2. 添加 OpenAI User-Agent",
            "headers": {
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI/Python 1.59.9",
            }
        },
        {
            "name": "3. 添加 X-Stainless 头 (OpenAI SDK特有)",
            "headers": {
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
                "X-Stainless-Lang": "python",
                "X-Stainless-Package-Version": "1.59.9",
                "X-Stainless-Runtime": "CPython",
                "X-Stainless-Runtime-Version": "3.11.2",
            }
        },
        {
            "name": "4. 完整 OpenAI SDK 请求头",
            "headers": {
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI/Python 1.59.9",
                "X-Stainless-Lang": "python",
                "X-Stainless-Package-Version": "1.59.9",
                "X-Stainless-Runtime": "CPython",
                "X-Stainless-Runtime-Version": "3.11.2",
                "X-Stainless-Arch": "x64",
                "X-Stainless-OS": "Linux",
            }
        },
        {
            "name": "5. 只添加 User-Agent: OpenAI",
            "headers": {
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI",
            }
        },
        {
            "name": "6. 模拟浏览器 User-Agent",
            "headers": {
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
        },
    ]

    print("=" * 70)
    print("测试不同请求头组合")
    print("=" * 70)
    print(f"API URL: {api_url}\n")

    async with httpx.AsyncClient(timeout=60) as client:
        for test in header_tests:
            print(f"{test['name']}")
            try:
                response = await client.post(api_url, headers=test["headers"], json=payload)
                if response.status_code == 200:
                    print(f"  ✅ 成功 (200)")
                else:
                    print(f"  ❌ 失败 ({response.status_code}): {response.text[:100]}")
            except Exception as e:
                print(f"  ❌ 异常: {type(e).__name__}: {e}")
            print()


async def test_openai_sdk_with_custom_headers(cfg: dict):
    """测试使用自定义 httpx client 的 OpenAI SDK"""

    print("=" * 70)
    print("测试: OpenAI SDK + 自定义 httpx client (移除问题头)")
    print("=" * 70)

    from openai import AsyncOpenAI

    # 创建自定义 httpx client，设置自定义 User-Agent
    custom_http_client = httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0",  # 覆盖默认的 OpenAI User-Agent
        }
    )

    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        http_client=custom_http_client,
    )

    try:
        print("  发送请求中...")
        completion = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Hello, reply with OK."}
            ],
            model=cfg.get("model", "gpt-4o"),
            temperature=0.1,
        )
        result = completion.choices[0].message.content.strip()
        print(f"  ✅ 请求成功!")
        print(f"  回答: {result}")
    except Exception as e:
        print(f"  ❌ 请求失败: {type(e).__name__}: {e}")
    finally:
        await custom_http_client.aclose()
    print()


async def test_openai_sdk_with_default_headers(cfg: dict):
    """测试使用默认 headers 覆盖的 OpenAI SDK"""

    print("=" * 70)
    print("测试: OpenAI SDK + default_headers 参数")
    print("=" * 70)

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        default_headers={
            "User-Agent": "Mozilla/5.0",
        }
    )

    try:
        print("  发送请求中...")
        completion = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Hello, reply with OK."}
            ],
            model=cfg.get("model", "gpt-4o"),
            temperature=0.1,
        )
        result = completion.choices[0].message.content.strip()
        print(f"  ✅ 请求成功!")
        print(f"  回答: {result}")
    except Exception as e:
        print(f"  ❌ 请求失败: {type(e).__name__}: {e}")
    print()


async def main():
    print("\n" + "=" * 70)
    print("OpenAI SDK 请求头诊断测试")
    print("=" * 70 + "\n")

    cfg = load_config()
    print(f"配置: {cfg.get('base_url')}, Model: {cfg.get('model')}\n")

    await test_headers(cfg)
    await test_openai_sdk_with_default_headers(cfg)
    await test_openai_sdk_with_custom_headers(cfg)

    print("=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
