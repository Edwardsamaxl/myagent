from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ...tools.registry import Tool
from .task_notification import NotificationType, TaskNotification
from .worker_result import WorkerResult

logger = logging.getLogger(__name__)


class Worker:
    """Worker executor with independent trace.

    Executes tasks and communicates with Coordinator via TaskNotification.
    """

    def __init__(
        self,
        worker_id: str,
        tools: dict[str, Tool],
        notify_callback: Callable[[TaskNotification], None] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.tools = tools
        self.notify_callback = notify_callback

    def execute(self, task: dict[str, Any]) -> WorkerResult:
        """Execute a task and return WorkerResult.

        Args:
            task: Task dict with keys:
                - task_id: str
                - task_type: str (e.g., "rag", "calc", "web")
                - input: str
                - context: dict[str, Any] (outputs from dependent tasks)

        Returns:
            WorkerResult with execution details.
        """
        task_id = task.get("task_id", self.worker_id)
        task_type = task.get("task_type", "unknown")
        task_input = task.get("input", "")
        context = task.get("context", {})

        start_time = time.time()

        try:
            self._send_notification(task_id, NotificationType.STATUS_UPDATE, {"status": "running"}, [])

            # Execute based on task type
            output = self._execute_task(task_type, task_input, context)

            latency_ms = int((time.time() - start_time) * 1000)

            self._send_notification(
                task_id,
                NotificationType.COMPLETED,
                WorkerResult(
                    worker_id=self.worker_id,
                    task_type=task_type,
                    output=output,
                    latency_ms=latency_ms,
                    success=True,
                ),
                deps_met=[task_id],
            )

            return WorkerResult(
                worker_id=self.worker_id,
                task_type=task_type,
                output=output,
                latency_ms=latency_ms,
                success=True,
            )

        except Exception as exc:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Worker {self.worker_id} execution failed: {exc}")

            self._send_notification(
                task_id,
                NotificationType.ERROR,
                WorkerResult(
                    worker_id=self.worker_id,
                    task_type=task_type,
                    output="",
                    latency_ms=latency_ms,
                    success=False,
                    error=str(exc),
                ),
                [],
            )

            return WorkerResult(
                worker_id=self.worker_id,
                task_type=task_type,
                output="",
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    def _execute_task(self, task_type: str, task_input: str, context: dict[str, Any]) -> str:
        """Execute task based on type using tools registry."""
        tool_map = {
            "rag": "search_knowledge_base",
            "calc": "calculate",
            "web": "web_search",
            "memory": "read_memory",
        }

        tool_name = tool_map.get(task_type)
        if not tool_name:
            # Build input from context for unknown types
            if context:
                inputs = [str(v) for v in context.values() if v]
                return f"Task output: {', '.join(inputs)}" if inputs else task_input
            return task_input

        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        # Build input: combine context outputs if dependencies exist
        if context:
            combined_input = "\n".join(str(v) for v in context.values() if v)
            combined_input = f"{combined_input}\n{task_input}".strip()
            tool_input = combined_input
        else:
            tool_input = self._clean_input(task_type, task_input)

        result = self.tools[tool_name].func(tool_input)
        return str(result)

    def _clean_input(self, task_type: str, task_input: str) -> str:
        """Remove common prefixes from task input."""
        prefixes: dict[str, tuple[str, ...]] = {
            "calc": ("计算 ", "执行 ", "求 ", "请计算", "运算 ", "calculate ", "compute ", "evaluate "),
            "web": ("搜索 ", "查询 ", "查找 ", "请搜索", "搜 ", "search ", "look up ", "find "),
            "rag": ("检索 ", "查询 ", "搜索 ", "查找 ", "搜 ", "search ", "retrieve ", "find "),
            "memory": ("读取", "查看", "获取", "read ", "get "),
        }

        prefixes_to_remove = prefixes.get(task_type, ())
        for p in prefixes_to_remove:
            if task_input.startswith(p):
                return task_input[len(p):].strip()
        return task_input

    def _send_notification(
        self,
        sender: str,
        notif_type: NotificationType,
        payload: Any,
        deps_met: list[str],
    ) -> None:
        """Send TaskNotification to Coordinator."""
        if self.notify_callback:
            notification = TaskNotification(
                sender=sender,
                type=notif_type,
                payload=payload,
                dependencies_met=deps_met,
            )
            self.notify_callback(notification)
