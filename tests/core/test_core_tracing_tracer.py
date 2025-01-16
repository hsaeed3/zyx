"""
Tests for zyx.core.tracing.tracer
"""

from zyx.core.tracing.tracer import _tracer, trace as tracing, on, emit
from zyx.core.tracing.events import TracedEventPattern
import pytest
from typing import Optional, Dict, Any, Literal


@pytest.fixture
def reset_tracer():
    """Fixture to reset the tracer state before each test."""
    _tracer.active_patterns.clear()
    _tracer.hooks.clear()
    _tracer.entity_map.clear()
    _tracer._context_stack.clear()
    yield
    _tracer.active_patterns.clear()
    _tracer.hooks.clear()
    _tracer.entity_map.clear()
    _tracer._context_stack.clear()


class MockAgent:
    """Mock agent class for testing tracing."""

    def __init__(self, name: str):
        self.name = name
        _tracer.register_entity(self.name, f"Agent({self.name})")
        with _tracer.module_context("agent", self.name):
            _tracer.log("create", {"name": name})

    def request_completion(self, prompt: str):
        with _tracer.module_context("agent", self.name):
            _tracer.log("completion:request", {"prompt": prompt})
            _tracer.log("completion:response", {"response": f"Mock response to: {prompt}"})

    def use_memory(self, data: str):
        with _tracer.module_context("agent", self.name):
            _tracer.log("memory:add", {"data": data})
            _tracer.log("memory:query", {"query": f"Finding {data}"})


class MockGraph:
    """Mock graph class for testing tracing."""

    def __init__(self, name: str):
        self.name = name
        with _tracer.module_context("graph"):
            _tracer.log("create", {"name": name})

    def add_agent(self, agent_name: str) -> MockAgent:
        with _tracer.module_context("graph"):
            agent = MockAgent(agent_name)
            _tracer.log("agent:create", {"name": agent_name})
            return agent

    def execute_action(self, action: str):
        with _tracer.module_context("graph"):
            _tracer.log("action:execute", {"action": action})


def test_agent_creation(reset_tracer):
    """Test agent creation events."""
    tracing("agent:create")
    agent = MockAgent("test_agent")
    assert "agent:create" in _tracer.active_patterns


def test_graph_with_agent(reset_tracer):
    """Test graph with agent interactions."""
    patterns = ["graph:agent:create", "graph:action:execute"]
    tracing(patterns)

    graph = MockGraph("test_graph")
    agent = graph.add_agent("test_agent")
    graph.execute_action("test_action")

    assert "graph:agent:create" in _tracer.active_patterns
    assert "graph:action:execute" in _tracer.active_patterns


def test_completion_events(reset_tracer):
    """Test completion request/response events."""
    patterns = ["agent:completion:request", "agent:completion:response"]
    tracing(patterns)

    agent = MockAgent("test_agent")
    agent.request_completion("Hello")

    assert "agent:completion:request" in _tracer.active_patterns
    assert "agent:completion:response" in _tracer.active_patterns


def test_memory_operations(reset_tracer):
    """Test memory operation events."""
    patterns = ["agent:memory:add", "agent:memory:query"]
    tracing(patterns)

    agent = MockAgent("test_agent")
    agent.use_memory("test_data")

    assert "agent:memory:add" in _tracer.active_patterns
    assert "agent:memory:query" in _tracer.active_patterns


def test_nested_event_patterns(reset_tracer):
    """Test nested event patterns with graph:agent:completion."""
    tracing("graph:agent:completion:request")

    graph = MockGraph("test_graph")
    agent = graph.add_agent("test_agent")
    agent.request_completion("Test nested events")

    assert "graph:agent:completion:request" in _tracer.active_patterns


def test_entity_specific_events(reset_tracer):
    """Test entity-specific event patterns."""
    tracing("agent(specific_agent):completion:request")

    agent1 = MockAgent("specific_agent")
    agent2 = MockAgent("other_agent")

    agent1.request_completion("Should trace")
    agent2.request_completion("Should not trace")

    pattern = "agent(specific_agent):completion:request"
    assert pattern in _tracer.active_patterns


def test_disable_all_tracing(reset_tracer):
    """Test disabling all tracing."""
    patterns = ["agent:create", "graph:create"]
    tracing(patterns)
    assert len(_tracer.active_patterns) > 0

    tracing(False)
    assert len(_tracer.active_patterns) == 0


def test_pattern_hooks_reject_all(reset_tracer):
    """Test that hooks reject 'all' patterns."""
    with pytest.raises(ValueError):

        @on("agent:all")
        def handle_all(data: Dict[str, Any]):
            pass

    with pytest.raises(ValueError):

        @on("all")
        def handle_everything(data: Dict[str, Any]):
            pass
