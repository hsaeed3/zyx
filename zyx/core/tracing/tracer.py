"""
### zyx.core.tracing.tracer

The `Tracer` class manages event tracing with support for:
1. Module-level tracing (e.g., 'completion:request')
2. Entity-specific tracing (e.g., 'steve:completion:request')
3. Dynamic entity renaming and resolution
4. Rich visualization via tracing_visualizer
"""

from __future__ import annotations

from typing import Callable, Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass
from contextlib import contextmanager
import re
from functools import wraps
import logging
import cachetools
import cattrs

from ..types.tracing import TraceContext, TracingStyles
from .events import (
    TracedModuleType,
    TracedEventType,
    TracedEventPattern,
    TRACED_MODULE_EVENTS,
    TRACED_MODULE_PATTERN,
    TRACED_EVENT_PATTERN,
    TRACED_MODULES,
)
from .visualizer import tracing_visualizer

logger = logging.getLogger("zyx.core.tracing")

# Configure cattrs for our dataclasses
try:
    converter = cattrs.Converter()
except Exception as e:
    logger.error(f"Failed to initialize cattrs converter: {e}")
    raise

# -----------------------------------------------------------------------------
# [Dataclasses & Types]
# -----------------------------------------------------------------------------


@dataclass
class ModuleSpec:
    """Module specification with optional entity."""

    module: str
    entity: Optional[str] = None

    def __str__(self) -> str:
        if self.entity:
            return f"{self.module}({self.entity})"
        return self.module


# -----------------------------------------------------------------------------
# Tracer
# -----------------------------------------------------------------------------


class Tracer:
    """Internal tracing system for zyx."""

    def __init__(self):
        """Initialize tracer with empty patterns and hook registry."""
        try:
            self.active_patterns: Set[str] = set()
            self.hooks: Dict[str, List[Callable[[str, Any], None]]] = {}
            self.entity_map: Dict[str, str] = {}
            self._context_stack: List[TraceContext] = []

            # Add caches for pattern matching
            self._pattern_cache = cachetools.TTLCache(maxsize=100, ttl=300)
            self._all_pattern_cache = cachetools.TTLCache(maxsize=100, ttl=300)

            tracing_visualizer.enable_live()
            logger.debug("CORE:TRACER initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Tracer: {e}")
            raise

    # -------------------------------------------------------------------------
    # [Pattern Parsing]
    # -------------------------------------------------------------------------

    def _check_all_pattern(self, pattern: str) -> bool:
        """Check if pattern uses 'all' wildcard."""
        try:
            return pattern in ("all", "True") or pattern.endswith(":all")
        except Exception as e:
            logger.error(f"Error checking all pattern: {e}")
            return False

    def _resolve_all_pattern(self, pattern: str) -> Set[str]:
        """Resolve 'all' pattern into concrete patterns."""
        try:
            if pattern in ("all", "True"):
                patterns = set()
                for module in TRACED_MODULES:
                    if module in TRACED_MODULE_EVENTS:
                        for event in TRACED_MODULE_EVENTS[module]:
                            patterns.add(f"{module}:{event}")
                return patterns

            if pattern.endswith(":all"):
                base = pattern[:-4]  # Remove ':all'
                if base in TRACED_MODULE_EVENTS:
                    return {f"{base}:{event}" for event in TRACED_MODULE_EVENTS[base]}

            return {pattern}
        except Exception as e:
            logger.error(f"Error resolving all pattern: {e}")
            return set()

    def _matches_pattern(self, pattern: str, event: str) -> bool:
        """Check if event matches pattern."""
        try:
            if pattern in ("all", "True"):
                return True

            pattern_parts = pattern.split(":")
            event_parts = event.split(":")

            if len(pattern_parts) > len(event_parts):
                return False

            for p_part, e_part in zip(pattern_parts, event_parts):
                if "(" in p_part:
                    p_module, p_entity = p_part[:-1].split("(")
                    if "(" in e_part:
                        e_module, e_entity = e_part[:-1].split("(")
                        if p_module != e_module or p_entity != e_entity:
                            return False
                    else:
                        return False
                elif p_part != e_part:
                    return False

            return True
        except Exception as e:
            logger.error(f"Error matching pattern: {e}")
            return False

    # -------------------------------------------------------------------------
    # [Hooks]
    # -------------------------------------------------------------------------

    def on(self, pattern: str) -> Callable:
        """Register hook for pattern."""

        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in hook for pattern '{pattern}': {e}")
                    raise

            try:
                # Validate pattern - reject 'all' patterns
                if self._check_all_pattern(pattern):
                    raise ValueError(
                        "Cannot register hooks with 'all' patterns. " "Please specify exact event patterns."
                    )

                if pattern not in self.hooks:
                    self.hooks[pattern] = []
                self.hooks[pattern].append(func)
                logger.debug(f"CORE:TRACER registered hook for '{pattern}'")
            except Exception as e:
                logger.error(f"Error registering hook for pattern '{pattern}': {e}")
                raise
            return wrapper

        return decorator

    def emit(self, event: str, data: Optional[Any] = None) -> Any:
        """Emit event to trigger hooks."""
        result = None
        try:
            # First check if we should process this event based on active patterns
            should_emit = any(self._matches_pattern(pattern, event) for pattern in self.active_patterns)

            if should_emit:
                # Then check hooks
                for pattern, handlers in self.hooks.items():
                    if self._matches_pattern(pattern, event):
                        for handler in handlers:
                            try:
                                result = handler(data) if data is not None else handler()
                                logger.debug(f"CORE:TRACER emitted {event} for pattern {pattern}")
                            except Exception as e:
                                logger.error(f"Error in handler for pattern '{pattern}': {e}")
        except Exception as e:
            logger.error(f"Error emitting event '{event}': {e}")
        return result

    # -------------------------------------------------------------------------
    # [Internal API]
    # -------------------------------------------------------------------------

    def register_entity(self, entity_id: str, display_name: str):
        """Register entity with display name."""
        try:
            self.entity_map[entity_id] = display_name
            logger.debug(f"CORE:TRACER registered entity '{entity_id}' as '{display_name}'")
        except Exception as e:
            logger.error(f"Error registering entity '{entity_id}': {e}")
            raise

    def log(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Internal logging method for modules."""
        try:
            ctx = self._context_stack[-1] if self._context_stack else None

            full_event = (
                f"{ctx.module}({ctx.entity}):{event}"
                if ctx and ctx.active and ctx.entity
                else f"{ctx.module}:{event}"
                if ctx and ctx.active
                else event
            )

            should_log = any(self._matches_pattern(pattern, full_event) for pattern in self.active_patterns)

            if should_log:
                parts = full_event.split(":")
                display = parts[-2] if len(parts) > 1 else parts[0]
                if "(" in display:
                    display = display.split("(")[1][:-1]

                tracing_visualizer.log(display, f"{full_event}: {data}")
                logger.debug(f"CORE:TRACER logged {full_event}")

            # Always emit the event, regardless of logging status
            self.emit(full_event, data)
        except Exception as e:
            logger.error(f"Error logging event '{event}': {e}")

    @contextmanager
    def module_context(self, module: str, entity: Optional[str] = None):
        """Context manager for module scope."""
        try:
            ctx = TraceContext(module=module, entity=entity, active=True)
            self._context_stack.append(ctx)
            try:
                yield ctx
            finally:
                self._context_stack.pop()
        except Exception as e:
            logger.error(f"Error in module context for '{module}': {e}")
            raise

    def trace(self, patterns: Union[str, List[str], bool]):
        """Enable/disable tracing visualization for pattern(s)."""
        try:
            if patterns is True:
                # Remove extra newlines and move print to visualizer
                tracing_visualizer.print_header()

            if patterns is False:
                self.active_patterns.clear()
                logger.debug("CORE:TRACER disabled all patterns")
                return

            # Convert to list of strings
            pattern_list = []
            if isinstance(patterns, str):
                pattern_list = [patterns]
            elif isinstance(patterns, list):
                pattern_list = [str(p) for p in patterns]

            for pattern in pattern_list:
                if self._check_all_pattern(pattern):
                    self.active_patterns.update(self._resolve_all_pattern(pattern))
                else:
                    self.active_patterns.add(pattern)

            logger.debug(f"CORE:TRACER enabled patterns: {patterns}")
        except Exception as e:
            logger.error(f"Error setting tracing patterns: {e}")
            raise


# -----------------------------------------------------------------------------
# [Singleton & Public API]
# -----------------------------------------------------------------------------

try:
    _tracer = Tracer()
except Exception as e:
    logger.error(f"Failed to initialize Tracer singleton: {e}")
    raise


def trace(patterns: Union[str, TracedEventPattern, List[Union[str, TracedEventPattern]], bool]):
    """Enable or disable tracing visualization for specific patterns.

    Args:
        patterns: The pattern(s) to enable tracing for. Can be:
            - A string pattern like "agent:completion:request"
            - A TracedEventPattern literal
            - A list of string patterns or TracedEventPattern literals
            - False to disable all tracing
            - "all" or "True" to enable all patterns
            - A pattern ending in ":all" to enable all events for that module

    Examples:
        # Enable tracing for all completion requests
        tracing("completion:request")

        # Enable tracing for a specific agent's completion requests
        tracing("agent(steve):completion:request")

        # Enable all tracing patterns
        tracing("all")

        # Enable all events for the completion module
        tracing("completion:all")

        # Disable all tracing
        tracing(False)
    """
    try:
        return _tracer.trace(patterns)
    except Exception as e:
        logger.error(f"Error in tracing function: {e}")
        raise


def on(pattern: Union[str, TracedEventPattern], func: Optional[Callable] = None) -> Union[Callable, None]:
    """Register a hook function to be called when a pattern matches.

    Can be used as either a decorator or a function.

    Args:
        pattern: The event pattern to match against. Cannot use 'all' patterns.
        func: Optional function to register as a hook. If not provided,
              returns a decorator.

    Returns:
        None if used as a function, or a decorator if used as a decorator.

    Raises:
        ValueError: If pattern contains 'all'

    Examples:
        # As a decorator
        @on("completion:request")
        def handle_request(data):
            print(f"Got request: {data}")

        # As a function
        def handle_response(data):
            print(f"Got response: {data}")
        on("completion:response", handle_response)
    """
    try:
        pattern_str = str(pattern)  # Convert pattern to string

        # Validate pattern - reject 'all' patterns
        if _tracer._check_all_pattern(pattern_str):
            raise ValueError("Cannot register hooks with 'all' patterns. " "Please specify exact event patterns.")

        if func is not None:
            if pattern_str not in _tracer.hooks:
                _tracer.hooks[pattern_str] = []
            _tracer.hooks[pattern_str].append(func)
            logger.debug(f"CORE:TRACER registered hook for '{pattern_str}'")
            return None
        return _tracer.on(pattern_str)
    except Exception as e:
        logger.error(f"Error in on function: {e}")
        raise


def emit(event: Union[str, TracedEventPattern], data: Optional[Any] = None) -> Any:
    """Emit an event to trigger registered hooks.

    Args:
        event: The event pattern to emit
        data: Optional data to pass to hook functions

    Returns:
        The return value from the last hook function that was called,
        or None if no hooks were triggered.

    Examples:
        # Emit an event with data
        emit("completion:request", {"prompt": "Hello"})

        # Emit an event without data
        emit("agent:create")
    """
    try:
        return _tracer.emit(event, data)
    except Exception as e:
        logger.error(f"Error in emit function: {e}")
        raise
