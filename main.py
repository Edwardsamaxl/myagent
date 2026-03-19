from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = str(CURRENT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agent.config import AgentConfig
from agent.web import create_app


def main() -> None:
    load_dotenv()
    config = AgentConfig.from_env()
    app = create_app(config)
    print("Mini Agent Web UI 已启动。")
    print(f"访问地址: http://{config.web_host}:{config.web_port}")
    print(f"当前模型: {config.model_provider}/{config.model_name}")
    app.run(host=config.web_host, port=config.web_port, debug=False)


if __name__ == "__main__":
    main()

