"""
MiniMax-M2.7 Tool-Calling 能力测试

用法:
  python scripts/test_tool_calling.py

测试内容:
  1. 基础 generate 调用是否正常
  2. Anthropic-compatible tool calling 格式是否支持
  3. OpenAI-compatible function calling 格式是否支持
"""

import os
import sys
import json
import requests

# 加载 .env
from dotenv import load_dotenv
load_dotenv()


ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "MiniMax-M2.7")


def test_basic_generate():
    """测试 1: 基础 generate 调用"""
    print("\n" + "="*60)
    print("测试 1: 基础 generate 调用")
    print("="*60)

    url = f"{ANTHROPIC_BASE_URL}/v1/messages"
    headers = {
        "Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "1+1等于几？请直接回答数字。"}]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [{}])
        text = content[0].get("text", "") if content else ""
        print(f"[PASS] 基础调用成功: {text[:100]}")
        return True
    except Exception as e:
        print(f"[FAIL] 基础调用失败: {e}")
        return False


def test_tool_calling_anthropic():
    """测试 2: Anthropic-compatible tool calling 格式"""
    print("\n" + "="*60)
    print("测试 2: Anthropic-compatible tool calling")
    print("="*60)

    url = f"{ANTHROPIC_BASE_URL}/v1/messages"
    headers = {
        "Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    tools = [
        {
            "name": "get_weather",
            "description": "获取城市天气",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
        "tools": tools
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        stop_reason = data.get("stop_reason", "")

        print(f"stop_reason: {stop_reason}")

        for item in content:
            if item.get("type") == "tool_use":
                print(f"[PASS] 模型返回 tool_use 调用:")
                print(f"  tool: {item.get('name')}")
                print(f"  input: {json.dumps(item.get('input', {}), ensure_ascii=False)}")
                return True
            elif item.get("type") == "text":
                print(f"  text: {item.get('text', '')[:200]}")

        print(f"[WARN] 没有返回 tool_use，可能不支持或格式不对")
        print(f"  stop_reason: {stop_reason}")
        print(f"  原始响应前500字符: {json.dumps(data, ensure_ascii=False)[:500]}")
        return False

    except Exception as e:
        print(f"[FAIL] Anthropic tool calling 测试失败: {e}")
        return False


def test_function_calling_openai():
    """测试 3: OpenAI-compatible function calling 格式"""
    print("\n" + "="*60)
    print("测试 3: OpenAI-compatible function calling")
    print("="*60)

    openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    # 如果没有配置 OpenAI API key，跳过
    if not openai_key or openai_key in ("", "your-api-key-here"):
        print("[SKIP] 未配置有效的 OPENAI_API_KEY，跳过 OpenAI function calling 测试")
        return None

    url = f"{openai_base}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
        "tools": tools
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])

        if choices and "message" in choices[0]:
            msg = choices[0]["message"]
            if "tool_calls" in msg:
                print(f"[PASS] 模型返回 tool_calls:")
                for tc in msg["tool_calls"]:
                    print(f"  tool: {tc.get('function', {}).get('name')}")
                    print(f"  args: {tc.get('function', {}).get('arguments')}")
                return True

        print(f"[WARN] 没有返回 tool_calls")
        print(f"  原始响应前500字符: {json.dumps(data, ensure_ascii=False)[:500]}")
        return False

    except Exception as e:
        print(f"[FAIL] OpenAI function calling 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("MiniMax-M2.7 Tool-Calling 能力测试")
    print("="*60)
    print(f"模型: {MODEL_NAME}")
    print(f"API: {ANTHROPIC_BASE_URL}")

    results = {}
    results["basic"] = test_basic_generate()
    results["anthropic_tc"] = test_tool_calling_anthropic()
    results["openai_fc"] = test_function_calling_openai()

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, ok in results.items():
        if ok is None:
            status = "SKIP"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")
