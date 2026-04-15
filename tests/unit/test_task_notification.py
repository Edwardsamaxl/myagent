"""Unit tests for TaskNotification protocol."""

from __future__ import annotations

import pytest
from agent.core.planning.task_notification import (
    NotificationType,
    TaskNotification,
)


class TestNotificationType:
    """Test NotificationType enum values."""

    def test_notification_types_exist(self):
        """All expected NotificationType values exist."""
        assert NotificationType.COMPLETED.value == "completed"
        assert NotificationType.STATUS_UPDATE.value == "status_update"
        assert NotificationType.DEPENDENCY_MET.value == "dependency_met"
        assert NotificationType.ERROR.value == "error"


class TestTaskNotification:
    """Test TaskNotification dataclass."""

    def test_create_basic_notification(self):
        """Create a basic notification."""
        notif = TaskNotification(
            sender="s1",
            type=NotificationType.COMPLETED,
        )
        assert notif.sender == "s1"
        assert notif.type == NotificationType.COMPLETED
        assert notif.payload is None

    def test_create_notification_with_payload(self):
        """Create notification with payload."""
        notif = TaskNotification(
            sender="s1",
            type=NotificationType.COMPLETED,
            payload={"result": "success", "data": [1, 2, 3]},
        )
        assert notif.payload == {"result": "success", "data": [1, 2, 3]}

    def test_create_notification_with_dependencies(self):
        """Create notification with dependency tracking."""
        notif = TaskNotification(
            sender="s3",
            type=NotificationType.DEPENDENCY_MET,
            dependencies_met=["s1", "s2"],
        )
        assert notif.dependencies_met == ["s1", "s2"]

    def test_to_dict_basic(self):
        """Serialize notification to dict."""
        notif = TaskNotification(
            sender="s1",
            type=NotificationType.COMPLETED,
            payload={"hits": 5},
        )
        d = notif.to_dict()
        assert d["sender"] == "s1"
        assert d["type"] == "completed"
        assert d["payload"] == {"hits": 5}
        assert d["dependencies_met"] == []

    def test_to_dict_with_dependencies(self):
        """Serialize notification with dependencies."""
        notif = TaskNotification(
            sender="s3",
            type=NotificationType.DEPENDENCY_MET,
            dependencies_met=["s1", "s2"],
        )
        d = notif.to_dict()
        assert d["dependencies_met"] == ["s1", "s2"]
        assert d["type"] == "dependency_met"

    def test_notification_type_is_string_enum(self):
        """NotificationType is a string enum for JSON serialization."""
        assert isinstance(NotificationType.COMPLETED, str)
        assert NotificationType.COMPLETED == "completed"

    def test_error_notification(self):
        """Create error notification."""
        notif = TaskNotification(
            sender="s2",
            type=NotificationType.ERROR,
            payload={"error": "connection timeout"},
        )
        d = notif.to_dict()
        assert d["type"] == "error"
        assert d["payload"]["error"] == "connection timeout"

    def test_status_update_notification(self):
        """Create status update notification."""
        notif = TaskNotification(
            sender="s1",
            type=NotificationType.STATUS_UPDATE,
            payload={"status": "running", "step": 1},
        )
        d = notif.to_dict()
        assert d["type"] == "status_update"
        assert d["payload"]["status"] == "running"
