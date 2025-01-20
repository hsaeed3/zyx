"""
Tests for the zyx.core.tracing system.
"""

import pytest
from typing import List, Dict, Any
from datetime import datetime

from zyx.core.tracing.tracer import tracer
from zyx.core.tracing.types import TracingReceiver, TracingEvent, Traceable


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestReceiver(TracingReceiver):
    """Test receiver that collects events"""

    def __init__(self):
        super().__init__()
        self.received_events: List[TracingEvent] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def update(self, event: TracingEvent) -> None:
        self.received_events.append(event)

    def clear(self) -> None:
        """Clear received events"""
        self.received_events.clear()


class Agent(Traceable):
    """Test agent class"""

    traced_events = {"create", "move", "interact"}

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        tracer.register_entity("agent", name)
        tracer.emit(f"agent({name}):create", {"name": name})

    def move_to(self, position: tuple[int, int]) -> None:
        tracer.emit(f"agent({self.name}):move", {"position": position})

    def interact_with(self, node_id: str) -> None:
        tracer.emit(f"agent({self.name}):interact", {"target": node_id})


class Graph(Traceable):
    """Test graph class"""

    traced_events = {"create", "add_node", "add_edge"}

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        tracer.register_entity("graph", name)
        tracer.emit(f"graph({name}):create", {"name": name})

    def add_node(self, node_id: str, data: dict) -> None:
        tracer.emit(f"graph({self.name}):add_node", {"node": node_id, "data": data})

    def add_edge(self, source: str, target: str) -> None:
        tracer.emit(f"graph({self.name}):add_edge", {"source": source, "target": target})


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_tracer():
    """Setup and cleanup tracer for each test"""
    # Clear existing state
    tracer.modules.clear()
    tracer.patterns.clear()
    tracer.listeners.clear()
    tracer.receivers.clear()
    tracer.active_receivers.clear()
    tracer.verbose_enabled = False
    tracer.verbose_enabled_all = False

    # Register core modules
    tracer.register_module("agent")
    tracer.register_module("graph")

    yield

    # Cleanup after test
    tracer.disable_all()


@pytest.fixture
def receiver():
    """Create and register a test receiver"""
    recv = TestReceiver()
    tracer.register_receiver("test", recv)
    tracer.enable_receiver("test")
    return recv


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_module_registration():
    """Test module registration"""
    tracer.register_module("test")
    assert "test" in tracer.modules

    # Duplicate registration
    tracer.register_module("test")
    assert len(tracer.modules) == 3  # test + agent/graph


def test_entity_registration():
    """Test entity registration"""
    # Valid registration
    tracer.register_entity("agent", "test")
    assert "agent(test)" in tracer.modules["agent"].listeners

    # Invalid module
    with pytest.raises(ValueError):
        tracer.register_entity("invalid", "test")


def test_pattern_matching():
    """Test pattern matching functionality"""
    tracer.trace(["agent(*):move"])
    assert tracer.is_pattern_enabled("agent(test):move")
    assert tracer.is_pattern_enabled("agent(other):move")
    assert not tracer.is_pattern_enabled("agent(test):interact")


def test_event_emission(receiver):
    """Test event emission and reception"""
    # Enable tracing
    tracer.trace(["agent(*)", "graph(*)"])

    # Create instances
    agent = Agent("test")
    graph = Graph("main")

    # Verify creation events
    assert len(receiver.received_events) == 2
    assert receiver.received_events[0].pattern == "agent(test):create"
    assert receiver.received_events[1].pattern == "graph(main):create"

    # Clear and test other events
    receiver.clear()

    agent.move_to((0, 0))
    graph.add_node("A", {"type": "test"})

    assert len(receiver.received_events) == 2
    assert "move" in receiver.received_events[0].pattern
    assert "add_node" in receiver.received_events[1].pattern


def test_listener_management():
    """Test listener registration and callback"""
    events = []

    def on_interact(event: TracingEvent):
        events.append(event)

    # Register listener
    tracer.on("agent(*):interact", on_interact)

    # Enable tracing
    tracer.trace(["agent(*):interact"])

    # Create agent and interact
    agent = Agent("test")
    agent.interact_with("node-1")

    assert len(events) == 1
    assert events[0].pattern == "agent(test):interact"
    assert events[0].data["target"] == "node-1"


def test_one_time_listener():
    """Test one-time listener behavior"""
    events = []

    def on_move(event: TracingEvent):
        events.append(event)

    # Register one-time listener
    tracer.on("agent(*):move", on_move, once=True)

    # Enable tracing
    tracer.trace(["agent(*):move"])

    # Create agent and move twice
    agent = Agent("test")
    agent.move_to((0, 0))
    agent.move_to((1, 1))

    # Should only capture first move
    assert len(events) == 1
    assert "(0, 0)" in str(events[0].data)


def test_receiver_management():
    """Test receiver registration and management"""
    recv1 = TestReceiver()
    recv2 = TestReceiver()

    # Register receivers
    tracer.register_receiver("recv1", recv1)
    tracer.register_receiver("recv2", recv2)

    # Enable only recv1
    tracer.enable_receiver("recv1")

    # Enable tracing and emit event
    tracer.trace(["test:event"])
    tracer.emit("test:event", {"data": "test"})

    # Only recv1 should receive
    assert len(recv1.received_events) == 1
    assert len(recv2.received_events) == 0

    # Enable recv2
    tracer.enable_receiver("recv2")
    tracer.emit("test:event", {"data": "test2"})

    # Both should receive
    assert len(recv1.received_events) == 2
    assert len(recv2.received_events) == 1


if __name__ == "__main__":
    pytest.main([__file__])
