from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mini_openclaw_agent.agent import SimpleAgent
from mini_openclaw_agent.config import AgentConfig
from mini_openclaw_agent.providers import build_model_provider
from mini_openclaw_agent.tools import default_tools


def main() -> None:
    load_dotenv()
    config = AgentConfig.from_env()
    model = build_model_provider(config)
    tools = default_tools()
    agent = SimpleAgent(config=config, model=model, tools=tools)

    print("mini-openclaw-agent 已启动，输入 exit 退出。")
    print(
        f"当前模型提供商: {config.model_provider} | 模型: {config.model_name} | 最大步骤: {config.max_steps}"
    )

    while True:
        user_input = input("\n你> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("再见。")
            break
        if not user_input:
            continue

        try:
            result = agent.run(user_input)
            print(f"\n助手({result.steps_used}步)> {result.answer}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n运行失败: {exc}")


if __name__ == "__main__":
    main()

