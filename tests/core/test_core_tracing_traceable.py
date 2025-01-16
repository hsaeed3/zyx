"""
### tests/core/tracing/test_traceable.py

Test suite for the Traceable abstract base class implementation.
Uses proper event types and patterns from zyx.core.tracing.events.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional

from zyx.core.tracing.traceable import Traceable
from zyx.core.tracing.tracer import _tracer
from zyx.core.tracing.events import TracedModuleType, TracedEventType, AgentTracedEventType, CompletionTracedEventType


class Agent(Traceable):
    """
    Test implementation of Traceable for an agent.
    Demonstrates proper usage of traced event types.
    """

    def __init__(self, name: str, parent: Optional[Traceable] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(name, parent=parent, metadata=metadata)
        self.state: Dict[str, Any] = {}

    def _get_trace_module(self) -> TracedModuleType:
        return "agent"

    def visualize(self):
        print(f"Visualizing agent: {self.name}")
        return f"Agent(name='{self.name}')"

    def send_completion(self, prompt: str) -> str:
        """Example method using proper completion event types."""
        with self.trace_context():
            self.log("completion:request", {"prompt": prompt, "metadata": self.metadata})
            response = f"Response to: {prompt}"
            self.log("completion:response", {"response": response, "metadata": self.metadata})
            return response


@pytest.fixture
def mock_tracer():
    """Fixture to mock the global tracer."""
    with patch("zyx.core.tracing.traceable._tracer") as mock:
        yield mock


def test_agent_initialization(mock_tracer):
    """Test agent initialization with proper create event."""
    agent = Agent("test_agent")
    mock_tracer.register_entity.assert_called_once_with("test_agent", "test_agent")
    mock_tracer.log.assert_called_with("create", {"name": "test_agent", "parent": None, "metadata": {}})
    mock_tracer.module_context.assert_called_with("agent", "test_agent")


def test_agent_rename(mock_tracer):
    """Test agent renaming with proper event logging."""
    agent = Agent("test_agent")
    mock_tracer.reset_mock()

    agent.name = "new_name"

    mock_tracer.register_entity.assert_called_once_with("test_agent", "new_name")
    mock_tracer.log.assert_called_with("rename", {"old_name": "test_agent", "new_name": "new_name"})


def test_agent_completion_flow(mock_tracer):
    """Test completion event flow with proper event types."""
    agent = Agent("test_agent")
    mock_tracer.reset_mock()

    response = agent.send_completion("Hello")

    mock_tracer.log.assert_any_call("completion:request", {"prompt": "Hello", "metadata": {}})
    mock_tracer.log.assert_any_call("completion:response", {"response": response, "metadata": {}})


def test_agent_parent_relationship(mock_tracer):
    """Test parent-child relationship with proper event types."""
    parent_agent = Agent("parent")
    child_agent = Agent("child", parent=parent_agent)
    mock_tracer.reset_mock()

    # Test parent change
    new_parent = Agent("new_parent")
    child_agent.parent = new_parent

    mock_tracer.log.assert_called_with("parent_change", {"old_parent": "parent", "new_parent": "new_parent"})


def test_agent_metadata_handling(mock_tracer):
    """Test metadata handling with proper event logging."""
    agent = Agent("test_agent", metadata={"role": "assistant"})
    mock_tracer.reset_mock()

    agent.add_metadata("temperature", 0.7)

    mock_tracer.log.assert_called_with("metadata_add", {"key": "temperature", "value": 0.7})

    # Test metadata in completion context
    response = agent.send_completion("Test")
    last_call_args = mock_tracer.log.call_args_list[-1][0]
    assert last_call_args[0] == "completion:response"
    assert last_call_args[1]["metadata"] == {"role": "assistant", "temperature": 0.7}


def test_agent_context_management(mock_tracer):
    """Test trace context management."""
    agent = Agent("test_agent")
    mock_tracer.reset_mock()

    with agent.trace_context() as ctx:
        assert ctx.module == "agent"
        assert ctx.entity == "test_agent"
        assert ctx.active is True

    mock_tracer.module_context.assert_called_once_with("agent", "test_agent")


def test_agent_error_handling(mock_tracer):
    """Test error handling during tracing operations."""
    agent = Agent("test_agent")
    mock_tracer.reset_mock()

    # Simulate tracer error
    mock_tracer.log.side_effect = Exception("Tracing error")

    with pytest.raises(Exception) as exc_info:
        agent.send_completion("Test")
    assert "Tracing error" in str(exc_info.value)


def test_integration_scenario(mock_tracer):
    """Test complex integration scenario with proper event patterns."""
    # Create parent agent
    parent = Agent("parent", metadata={"type": "supervisor"})
    mock_tracer.reset_mock()

    # Create child agent
    child = Agent("child", parent=parent, metadata={"type": "worker"})

    # Test completion flow with parent-child relationship
    child.send_completion("Hello")

    # Verify proper module contexts were used
    mock_tracer.module_context.assert_called_with("agent", "child")

    # Verify proper event patterns in logs
    logged_events = [args[0][0] for args in mock_tracer.log.call_args_list]
    assert "completion:request" in logged_events
    assert "completion:response" in logged_events

    # Verify metadata propagation
    last_call = mock_tracer.log.call_args_list[-1]
    assert last_call[0][1]["metadata"]["type"] == "worker"


if __name__ == "__main__":
    # enable printing
    pytest.main(args=["-s"])
