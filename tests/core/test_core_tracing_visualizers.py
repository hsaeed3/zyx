"""
Tests for zyx.core.tracing.visualizers
"""

from zyx.core.tracing.visualizer import tracing_visualizer
from rich.spinner import Spinner
import pytest


@pytest.fixture
def reset_visualizer():
    """Fixture to reset the visualizer state before each test."""
    tracing_visualizer.disable_live()
    tracing_visualizer._logs.clear()
    tracing_visualizer._tasks.clear()
    yield
    tracing_visualizer.disable_live()
    tracing_visualizer._logs.clear()
    tracing_visualizer._tasks.clear()


def test_enable_disable_live(reset_visualizer):
    """Test enabling and disabling live tracing."""
    tracing_visualizer.enable_live()
    assert tracing_visualizer._is_active, "Live tracing should be active after enabling."

    tracing_visualizer.disable_live()
    assert not tracing_visualizer._is_active, "Live tracing should be inactive after disabling."


def test_log_messages(reset_visualizer):
    """Test logging messages to the visualizer."""
    tracing_visualizer.enable_live()

    tracing_visualizer.log("test_entity", "This is a test message")
    tracing_visualizer.log("test_entity", "This is another test message")

    assert "test_entity" in tracing_visualizer._logs, "Entity should exist in logs after adding messages."
    assert len(tracing_visualizer._logs["test_entity"].children) == 2, "Entity should have two logged messages."


def test_add_spinner_task(reset_visualizer):
    """Test adding a spinner task to the visualizer."""
    tracing_visualizer.enable_live()

    tracing_visualizer.add_task("test_entity", "Loading something")
    assert "Loading something" in tracing_visualizer._tasks, "Spinner task should be added to tasks."

    spinner = tracing_visualizer._tasks["Loading something"]
    assert isinstance(spinner, Spinner), "Task should be a Spinner instance."


def test_add_progress_task(reset_visualizer):
    """Test adding a progress bar task to the visualizer."""
    tracing_visualizer.enable_live()

    tracing_visualizer.add_task("test_entity", "Processing data", total=10)
    assert "Processing data" in tracing_visualizer._tasks, "Progress bar task should be added to tasks."

    progress_task = tracing_visualizer._tasks["Processing data"]
    assert isinstance(progress_task, tuple), "Task should be a tuple for progress bars."
    assert len(progress_task) == 2, "Progress task should contain Progress and task_id."


def test_update_progress_task(reset_visualizer):
    """Test updating a progress bar task."""
    tracing_visualizer.enable_live()

    tracing_visualizer.add_task("test_entity", "Processing data", total=10)
    tracing_visualizer.update_task("Processing data", advance=5)

    progress, task_id = tracing_visualizer._tasks["Processing data"]
    task = progress.tasks[task_id]
    assert task.completed == 5, "Progress task should have advanced by 5 steps."


def test_complete_task(reset_visualizer):
    """Test completing and removing a task."""
    tracing_visualizer.enable_live()

    tracing_visualizer.add_task("test_entity", "Finalizing")
    tracing_visualizer.complete_task("Finalizing")

    assert "Finalizing" not in tracing_visualizer._tasks, "Task should be removed after completion."
