from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from ...config import AgentConfig
from ...llm.providers import ModelProvider
from ...tools.registry import Tool
from .synthesizer import Synthesizer
from .task_notification import NotificationType, TaskNotification
from .worker import Worker
from .worker_result import WorkerResult

logger = logging.getLogger(__name__)


# RAG priority constant
ACTION_PRIORITY = {
    "rag": 0,
    "calc": 1,
    "web": 1,
    "memory": 1,
    "synthesize": 2,
}


@dataclass
class CoordinatorResult:
    """Result of Coordinator execution."""
    answer: str                                    # Final response
    plan_id: str                                   # Plan identifier
    worker_results: dict[str, WorkerResult]       # Per-worker results
    total_tasks: int                              # Total tasks executed
    tool_calls: list[str]                          # Tool call records


class Coordinator:
    """Coordinator for multi-agent task execution.

    Responsibilities (separated from Planner):
    - Assign tasks to Workers
    - Collect TaskNotifications
    - Call Synthesizer for final summary

    Does NOT handle:
    - Task planning/decomposition (handled by LLM Router)
    """

    def __init__(
        self,
        config: AgentConfig,
        model: ModelProvider,
        tools: dict[str, Tool],
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools
        self.synthesizer = Synthesizer(model, config)

        # RAG mutex: ensure only one RAG call
        self._rag_lock = threading.Lock()
        self._rag_completed_hits: list[dict] = []
        self._rag_executed: bool = False

        # Worker registry
        self._workers: dict[str, Worker] = {}

    def _handle_notification(self, notification: TaskNotification) -> None:
        """Handle TaskNotification from Workers (for observability logging)."""
        logger.info(
            f"[TaskNotification] from={notification.sender} "
            f"type={notification.type.value} deps={notification.dependencies_met}"
        )

    def register_worker(self, worker_id: str) -> Worker:
        """Register a worker and return it."""
        if worker_id not in self._workers:
            self._workers[worker_id] = Worker(
                worker_id=worker_id,
                tools=self.tools,
                notify_callback=self._handle_notification,
            )
        return self._workers[worker_id]

    def run(
        self,
        tasks: list[dict[str, Any]],
        user_input: str,
        plan_id: str = "",
        rag_hits: list[dict] | None = None,
    ) -> CoordinatorResult:
        """Execute tasks and generate final response.

        Args:
            tasks: List of task dicts with keys:
                - task_id: str
                - task_type: str (e.g., "rag", "calc")
                - input: str
                - depends_on: list[str] (optional, task_ids this depends on)
            user_input: Original user query.
            plan_id: Plan identifier for tracking.
            rag_hits: Optional pre-fetched RAG hits.

        Returns:
            CoordinatorResult with final answer and execution metadata.
        """
        rag_hits = rag_hits or []

        # Pre-populate RAG hits if provided
        if rag_hits:
            self._rag_completed_hits.extend(rag_hits)
            self._rag_executed = True

        # Execute tasks
        worker_results = self._execute_tasks(tasks)

        # Synthesize final response
        answer = self.synthesizer.synthesize(user_input, worker_results, rag_hits=rag_hits)

        # Collect tool calls
        tool_calls = [
            f"{r.task_type}:{r.worker_id}"
            for r in worker_results.values()
            if r.success
        ]

        return CoordinatorResult(
            answer=answer,
            plan_id=plan_id,
            worker_results=worker_results,
            total_tasks=len(tasks),
            tool_calls=tool_calls,
        )

    def _execute_tasks(self, tasks: list[dict[str, Any]]) -> dict[str, WorkerResult]:
        """Execute tasks respecting dependencies, with parallel execution where possible."""
        results: dict[str, WorkerResult] = {}
        pending = {t["task_id"]: t for t in tasks}
        max_iterations = len(tasks) * 2

        for _ in range(max_iterations):
            if not pending:
                break

            # Get tasks ready to execute (dependencies satisfied)
            ready = self._get_ready_tasks(pending, results)

            if not ready:
                # Cannot proceed - mark remaining as failed
                for task_id, task in pending.items():
                    results[task_id] = WorkerResult(
                        worker_id=task_id,
                        task_type=task.get("task_type", "unknown"),
                        output="",
                        latency_ms=0,
                        success=False,
                        error="Dependency failed or circular dependency",
                    )
                pending.clear()
                break

            # Group by type for RAG mutex handling
            rag_tasks = [t for t in ready if t.get("task_type") == "rag"]
            other_tasks = [t for t in ready if t.get("task_type") != "rag"]

            # Build context from completed results
            def build_context(task: dict) -> dict[str, Any]:
                context = {}
                for dep_id in task.get("depends_on", []):
                    if dep_id in results:
                        context[dep_id] = results[dep_id].output
                return context

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}

                # RAG tasks: use lock to ensure mutual exclusion
                for task in rag_tasks:
                    worker = self.register_worker(task["task_id"])
                    task_with_context = {
                        **task,
                        "context": build_context(task),
                    }
                    futures[executor.submit(self._execute_rag_with_lock, worker, task_with_context)] = task

                # Non-RAG tasks: parallel execution
                for task in other_tasks:
                    worker = self.register_worker(task["task_id"])
                    task_with_context = {
                        **task,
                        "context": build_context(task),
                    }
                    futures[executor.submit(worker.execute, task_with_context)] = task

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        results[task["task_id"]] = result
                    except Exception as exc:
                        results[task["task_id"]] = WorkerResult(
                            worker_id=task["task_id"],
                            task_type=task.get("task_type", "unknown"),
                            output="",
                            latency_ms=0,
                            success=False,
                            error=str(exc),
                        )
                    del pending[task["task_id"]]

        return results

    def _execute_rag_with_lock(
        self,
        worker: Worker,
        task: dict[str, Any],
    ) -> WorkerResult:
        """Execute RAG task with mutex lock to ensure only one RAG call."""
        with self._rag_lock:
            # If RAG already executed, reuse results
            if self._rag_executed and self._rag_completed_hits:
                from ...core.evidence_format import format_evidence_block_from_api_dicts

                output = format_evidence_block_from_api_dicts(self._rag_completed_hits)
                return WorkerResult(
                    worker_id=task["task_id"],
                    task_type="rag",
                    output=output,
                    latency_ms=0,
                    success=True,
                    metadata={"rag_hits_shared": True, "chunk_count": len(self._rag_completed_hits)},
                )

            # Execute RAG
            result = worker.execute(task)

            if result.success and result.metadata and result.metadata.get("rag_hits"):
                self._rag_completed_hits.extend(result.metadata["rag_hits"])
                self._rag_executed = True

            return result

    def _get_ready_tasks(
        self,
        pending: dict[str, dict[str, Any]],
        results: dict[str, WorkerResult],
    ) -> list[dict[str, Any]]:
        """Return tasks whose dependencies are all satisfied, sorted by priority."""
        ready = []
        for task_id, task in pending.items():
            deps_met = all(
                dep in results and results[dep].success
                for dep in task.get("depends_on", [])
            )
            if deps_met:
                ready.append(task)
        return sorted(ready, key=lambda t: ACTION_PRIORITY.get(t.get("task_type", ""), 99))
