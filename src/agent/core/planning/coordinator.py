from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from ...config import AgentConfig
from ...llm.providers import ModelProvider
from ...tools.registry import Tool
from .plan_schema import PlanArtifact, PlanStep, PlanStepAction
from .synthesizer import Synthesizer
from .worker_result import WorkerResult

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家。
分析用户问题，将其分解为可并行/串行执行的步骤。

可用动作：
- rag: 在知识库中检索相关文档
- calc: 执行算术计算
- web: 在互联网上搜索信息
- memory: 读写长期记忆
- synthesize: 汇总结果生成最终回复

分解原则：
1. 独立任务可以并行执行
2. 有依赖的任务必须串行（后续步骤依赖前序步骤的结果）
3. 每个步骤应该单一职责
4. 简单问题可以直接 synthesize，不需要拆解
5. 检索和计算可以并行，最后 synthesize

输出格式（仅输出 JSON，不要有其他内容）：
{
  "plan_id": "p_时间戳_随机6位",
  "goal": "用户问题的简洁描述",
  "steps": [
    {"id": "s1", "action": "rag", "detail": "...", "depends_on": [], "parallel_with": []},
    {"id": "s2", "action": "calc", "detail": "...", "depends_on": [], "parallel_with": ["s1"]},
    {"id": "s3", "action": "synthesize", "detail": "...", "depends_on": ["s1", "s2"], "parallel_with": []}
  ]
}
"""


class Planner:
    """任务分解器：将用户问题分解为可执行的步骤计划"""

    def __init__(self, model: ModelProvider, config: AgentConfig) -> None:
        self.model = model
        self.config = config

    def create_plan(self, user_input: str, long_context: str = "") -> PlanArtifact:
        """调用 AI 生成任务分解计划"""

        messages: list[dict] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        ]

        if long_context:
            messages.append({
                "role": "system",
                "content": f"长期上下文（记忆与技能）：\n{long_context}"
            })

        messages.append({
            "role": "user",
            "content": f"请分解这个任务：{user_input}"
        })

        try:
            response = self.model.generate(
                messages=messages,
                temperature=0.3,  # 规划时温度不宜过高
                max_tokens=768,
            )

            # 解析 JSON 返回（可能包含 markdown fence）
            raw = response.strip()
            if raw.startswith("```"):
                # 去掉 ```json ... ``` 包装
                parts = raw.split("```", 2)
                if len(parts) >= 3:
                    raw = parts[1].strip()
                    # 去掉第一行的 json 标识
                    lines = raw.split("\n", 1)
                    if lines and lines[0].strip() == "json":
                        raw = lines[1].strip() if len(lines) > 1 else ""
            plan_data = json.loads(raw)
            return self._parse_plan(plan_data)
        except json.JSONDecodeError:
            logger.warning(f"Planner JSON 解析失败，降级为直接 synthesize: {response[:100]}")
            return self._fallback_plan(user_input)
        except Exception as exc:
            logger.error(f"Planner 执行失败: {exc}")
            return self._fallback_plan(user_input)

    # action 别名映射：模型可能返回变体名称
    _ACTION_ALIASES: dict[str, PlanStepAction] = {
        "retrieve_financial_data": PlanStepAction.RAG,
        "query_data": PlanStepAction.RAG,
        "query": PlanStepAction.RAG,
        "retrieve": PlanStepAction.RAG,
        "search": PlanStepAction.WEB,
        "synthesize": PlanStepAction.SYNTHESIZE,
        "summarize": PlanStepAction.SYNTHESIZE,
        "math": PlanStepAction.CALC,
    }

    def _parse_plan(self, plan_data: dict) -> PlanArtifact:
        """解析 AI 返回的 JSON 为 PlanArtifact"""
        steps = []
        for s in plan_data.get("steps", []):
            raw_action = s.get("action", "")
            try:
                action = PlanStepAction(raw_action)
            except ValueError:
                # 尝试别名映射
                action = self._ACTION_ALIASES.get(raw_action.lower(), None)
                if action is None:
                    logger.warning(f"未知的 action 类型: {raw_action}，跳过该步骤")
                    continue
            step = PlanStep(
                id=s["id"],
                action=action,
                detail=s["detail"],
                depends_on=s.get("depends_on", []),
                parallel_with=s.get("parallel_with", []),
            )
            steps.append(step)

        return PlanArtifact(
            plan_id=plan_data.get("plan_id", f"p_{uuid.uuid4().hex[:6]}"),
            goal=plan_data.get("goal", ""),
            steps=steps,
        )

    def _fallback_plan(self, user_input: str) -> PlanArtifact:
        """降级计划：简单问题直接 synthesize"""
        return PlanArtifact(
            plan_id=f"p_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
            goal=user_input,
            steps=[
                PlanStep(
                    id="s1",
                    action=PlanStepAction.SYNTHESIZE,
                    detail="直接回答用户问题",
                )
            ],
        )


class WorkerExecutor:
    """Worker 执行器：根据 action 类型调用对应的工具"""

    def __init__(self, tools: dict[str, Tool]) -> None:
        self.tools = tools

    def execute(self, step: PlanStep, context: dict[str, WorkerResult]) -> WorkerResult:
        """执行单个步骤，返回结果"""

        # SYNTHESIZE 步骤不需要工具，由 Synthesizer 直接处理
        if step.action == PlanStepAction.SYNTHESIZE:
            return WorkerResult(
                step_id=step.id,
                action=step.action,
                status="success",
                output=step.detail,
            )

        action_to_tool = {
            PlanStepAction.RAG: "search_knowledge_base",
            PlanStepAction.CALC: "calculate",
            PlanStepAction.WEB: "web_search",
            PlanStepAction.MEMORY: "read_memory",
        }

        tool_name = action_to_tool.get(step.action)
        if not tool_name or tool_name not in self.tools:
            return WorkerResult(
                step_id=step.id,
                action=step.action,
                status="failed",
                output="",
                error=f"工具不存在: {tool_name}",
            )

        # 对于需要前序结果的步骤，从 context 获取输入
        tool_input = self._build_input(step, context)

        try:
            result = self.tools[tool_name].func(tool_input)
            return WorkerResult(
                step_id=step.id,
                action=step.action,
                status="success",
                output=str(result),
            )
        except Exception as exc:
            return WorkerResult(
                step_id=step.id,
                action=step.action,
                status="failed",
                output="",
                error=str(exc),
            )

    # 工具输入前缀清理（当无依赖时）
    _INPUT_PREFIXES: dict[str, tuple[str, ...]] = {
        "calc": (
            "计算 ", "执行 ", "求 ", "请计算", "运算 ",
            "calculate ", "compute ", "evaluate ", "what is ", "calc "
        ),
        "web": (
            "搜索 ", "查询 ", "查找 ", "请搜索", "搜 ",
            "search ", "look up ", "find ", "search for "
        ),
        "rag": (
            "检索 ", "查询 ", "搜索 ", "查找 ", "搜 ",
            "search ", "retrieve ", "find "
        ),
        "memory": ("读取", "查看", "获取", "read ", "get "),
    }

    def _build_input(self, step: PlanStep, context: dict[str, WorkerResult]) -> str:
        """根据依赖构建工具输入"""
        if step.depends_on:
            # 有依赖：组合依赖输出
            inputs = []
            for dep_id in step.depends_on:
                if dep_id in context and context[dep_id].is_success():
                    inputs.append(context[dep_id].output)
            return "\n".join(inputs) if inputs else step.detail

        # 无依赖：清理 detail 开头的前缀
        detail = step.detail
        prefixes = self._INPUT_PREFIXES.get(step.action.value, ())
        for p in prefixes:
            if detail.startswith(p):
                detail = detail[len(p):].strip()
                break
        return detail


@dataclass
class CoordinatorResult:
    """Coordinator 执行结果"""
    answer: str                       # 最终回复
    plan_id: str                      # 计划 ID
    steps: list[PlanStep]             # 计划步骤列表
    worker_results: dict[str, WorkerResult]  # 各 worker 的执行结果
    total_steps: int                 # 总步骤数
    tool_calls: list[str]            # 所有工具调用记录


class Coordinator:
    """Coordinator：主调度器，管理任务分解和执行流程"""

    def __init__(
        self,
        config: AgentConfig,
        model: ModelProvider,
        tools: dict[str, Tool],
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools
        self.planner = Planner(model, config)
        self.synthesizer = Synthesizer(model, config)
        self.worker_executor = WorkerExecutor(tools)

    def run(
        self,
        user_input: str,
        history_messages: list[dict] | None = None,
        long_context: str = "",
    ) -> CoordinatorResult:
        """执行完整的 Coordinator 流程"""

        history_messages = history_messages or []

        # Step 1: 任务分解
        plan = self.planner.create_plan(user_input, long_context)

        # Step 2: 执行计划
        worker_results = self._execute_plan(plan)

        # Step 3: 汇总生成
        answer = self.synthesizer.synthesize(user_input, plan, worker_results)

        # 收集所有工具调用
        tool_calls = [
            f"{r.action.value}:{r.step_id}"
            for r in worker_results.values()
            if r.is_success()
        ]

        return CoordinatorResult(
            answer=answer,
            plan_id=plan.plan_id,
            steps=plan.steps,
            worker_results=worker_results,
            total_steps=len(plan.steps),
            tool_calls=tool_calls,
        )

    def _execute_plan(self, plan: PlanArtifact) -> dict[str, WorkerResult]:
        """执行计划：处理依赖关系，并行/串行调度"""

        results: dict[str, WorkerResult] = {}
        pending = {s.id: s for s in plan.steps}
        max_iterations = len(plan.steps) * 2  # 防止死循环

        for _ in range(max_iterations):
            if not pending:
                break

            # 找出所有依赖已满足且未执行的步骤
            ready = []
            for step_id, step in pending.items():
                deps_met = all(
                    dep in results and results[dep].is_success()
                    for dep in step.depends_on
                )
                if deps_met:
                    ready.append(step)

            if not ready:
                # 无法继续执行（可能是循环依赖或全部失败）
                # 将剩余步骤标记为 failed
                for step_id, step in pending.items():
                    results[step_id] = WorkerResult(
                        step_id=step_id,
                        action=step.action,
                        status="failed",
                        output="",
                        error="依赖步骤执行失败或循环依赖",
                    )
                pending.clear()
                break

            # 并行执行所有就绪的步骤（ThreadPoolExecutor 实现真正并行）
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.worker_executor.execute, step, results): step
                    for step in ready
                }
                for future in as_completed(futures):
                    step = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = WorkerResult(
                            step_id=step.id,
                            action=step.action,
                            status="failed",
                            output="",
                            error=str(exc),
                        )
                    results[step.id] = result
                    del pending[step.id]

        return results
