"""
### zyx.core.tracing.types

Contains trace-related types & simple data structures:
  - Traceable
  - TracingReceiver 
  - TracingPattern
  - TracingEvent
  - TracingListener
  - TracingRegistry

Example:
    ```python
    # Create a traceable class
    class MyAgent(Traceable):
        traced_events = {"move", "interact"}
        
    # Create a receiver
    class ConsoleReceiver(TracingReceiver):
        def update(self, event):
            print(f"Received: {event}")
            
    # Create and match patterns
    pattern = TracingPattern.parse("agent(bob):move")
    pattern.matches("agent(*):*") # True
    
    # Create events and listeners
    event = TracingEvent(pattern="agent(bob):move", data={"x": 0, "y": 0})
    listener = TracingListener("agent(*):move", lambda e: print(e))
    
    # Use registry to store events/listeners
    registry = TracingRegistry()
    registry.events["move"] = event
    registry.listeners["move"] = listener
    ```

Args:
    None - this is a module containing type definitions

Returns:
    None - this is a module containing type definitions
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Union,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# [Traceable]
# -----------------------------------------------------------------------------
class Traceable(ABC):
    """
    A base class for objects or modules that can be traced by the system.
    """

    # Note: class-level sets are shared across subclasses unless re-declared.
    # You could also define them per instance with __init__ if needed.
    traced_events: Set[str] = set()
    traced_listeners: Set[TracingListener] = set()

    @classmethod
    def register_event(cls, event_name: str) -> None:
        """
        Register a new traced event name with this class.
        """
        cls.traced_events.add(event_name)

    @classmethod
    def register_entity(cls, entity_name: str) -> None:
        """
        Register a new 'entity' (e.g., sub-id) with this class.
        """
        if not hasattr(cls, "traced_entities"):
            cls.traced_entities = set()
        cls.traced_entities.add(entity_name)


# -----------------------------------------------------------------------------
# [TracingReceiver]
# -----------------------------------------------------------------------------
class TracingReceiver(ABC):
    """
    A base class for any object that receives traced events (e.g., console, UI).
    """

    def __init__(self) -> None:
        self.listeners: Dict[str, TracingListener] = {}
        self.events: Dict[str, TracingEvent] = {}

    @abstractmethod
    def update(self, event: TracingEvent) -> None:
        """
        Called when a new event is emitted.
        """
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """
        Called to initialize or 'start' the receiver.
        """
        raise NotImplementedError

    def add_listener(self, listener: TracingListener) -> None:
        """
        Add a listener to the receiver.
        """
        if listener.pattern in self.listeners:
            logger.warning(f"Listener already exists: {listener.pattern}")
            return
        self.listeners[listener.pattern] = listener

    def remove_listener(self, pattern: str) -> None:
        """
        Remove a listener by pattern.
        """
        if pattern not in self.listeners:
            logger.warning(f"No listener for pattern: {pattern}")
            return
        del self.listeners[pattern]

    def add_event(self, event: TracingEvent) -> None:
        """
        Store or track an event by pattern.
        """
        if event.pattern in self.events:
            logger.warning(f"Event already exists: {event.pattern}")
            return
        self.events[event.pattern] = event

    def remove_event(self, pattern: str) -> None:
        """
        Remove a tracked event by pattern.
        """
        if pattern not in self.events:
            logger.warning(f"No event for pattern: {pattern}")
            return
        del self.events[pattern]


# -----------------------------------------------------------------------------
# [TracingPattern]
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TracingPattern:
    """
    A pattern for matching or filtering traced events.
    Format typically: 'module', 'module(entity)', or 'module(entity):event'.
    """

    module: str
    entity: Optional[Union[str, list[str]]] = None
    event: Optional[str] = None

    @staticmethod
    def _validate_parentheses(pattern: str) -> None:
        """
        Ensure parentheses in the pattern are balanced, or raise ValueError.
        """
        open_count = pattern.count("(")
        close_count = pattern.count(")")
        if open_count != close_count:
            raise ValueError(f"Mismatched parentheses: {pattern}")

    @staticmethod
    def _split_module_event(pattern: str) -> tuple[str, Optional[str]]:
        """
        Split 'module:evt' -> ('module', 'evt'), or just return (pattern, None).
        """
        if ":" in pattern:
            mod, evt = pattern.split(":", 1)
            return mod, evt
        return pattern, None

    @staticmethod
    def _extract_entity(module_part: str) -> Optional[Union[str, list[str]]]:
        """
        Extract entity substring(s) from the module part (within (...)).
        Returns None if no entity is found.
        """
        matches = re.findall(r"\((.*?)\)", module_part)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # Multiple entity usage is not "fully" standardized, but we handle it
        return matches

    @classmethod
    @lru_cache(None)
    def parse(cls, pattern: str) -> TracingPattern:
        """
        Parse a pattern string, returning a TracingPattern instance.
        Uses caching to speed up repeated pattern parsing.
        """
        if not pattern or not isinstance(pattern, str):
            raise ValueError("Pattern must be a non-empty string.")

        cls._validate_parentheses(pattern)
        mod_part, evt = cls._split_module_event(pattern)
        ent = cls._extract_entity(mod_part)

        # If we have an entity, strip out the parentheses from 'module' substring
        if ent is not None:
            # module_part is everything before '('
            mod_part = mod_part.split("(", 1)[0]

        logger.debug(f"Parsed pattern => module: {mod_part}, entity: {ent}, event: {evt}")
        return cls(module=mod_part, entity=ent, event=evt)

    def matches(self, other: Union[str, TracingPattern]) -> bool:
        """
        Compare this pattern with another pattern/string for a match.
        Matching rules:
          - module must be exactly the same unless one side is '*'.
          - entity must match if both sides specify one (unless '*' is used).
          - event must match if both sides specify one (unless 'all'/'*' is used).
        """
        if isinstance(other, str):
            try:
                other = TracingPattern.parse(other)
            except ValueError:
                return False

        # Compare modules
        if self.module != other.module and self.module != "*" and other.module != "*":
            return False

        # Compare entities if both have them
        if self.entity and other.entity:
            if self.entity != other.entity and self.entity != "*" and other.entity != "*":
                return False

        # Compare events if both sides specify them
        if self.event and other.event:
            if self.event not in ("all", "*") and other.event not in ("all", "*"):
                if self.event != other.event:
                    return False

        return True

    def __str__(self) -> str:
        """
        Convert pattern back to a single string representation.
        """
        base = self.module
        if self.entity:
            # If entity is a list, we just place them in parentheses joined by commas
            if isinstance(self.entity, list):
                base += f"({','.join(self.entity)})"
            else:
                base += f"({self.entity})"
        if self.event:
            return f"{base}:{self.event}"
        return base


# -----------------------------------------------------------------------------
# [TracingEvent]
# -----------------------------------------------------------------------------
@dataclass
class TracingEvent:
    """
    A simple container for event data.
    """

    is_enabled: bool = False
    is_update: bool = False
    pattern: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# [TracingListener]
# -----------------------------------------------------------------------------
@dataclass
class TracingListener:
    """
    A callback listener that triggers on certain patterns.
    """

    pattern: str
    callback: Callable[[TracingEvent], None] = field(
        default_factory=lambda: (lambda e: None)
    )
    once: bool = False


# -----------------------------------------------------------------------------
# [TracingRegistry]
# -----------------------------------------------------------------------------
@dataclass
class TracingRegistry:
    """
    A registry storing events & listeners for a specific module or entity.
    """

    events: Dict[str, TracingEvent] = field(default_factory=dict)
    listeners: Dict[str, TracingListener] = field(default_factory=dict)
