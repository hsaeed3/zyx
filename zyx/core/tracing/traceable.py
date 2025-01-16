"""
### zyx.core.tracing.traceable

Contains the `Traceable` class, an abstract base class for entities or modules in zyx,
that support being dynamically renamed by the user, as well as may contain hierarchical
tracing in both directions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator
import logging

from .tracer import _tracer
from ..types.tracing import TraceContext

logger = logging.getLogger("zyx.core.tracing")


# -----------------------------------------------------------------------------
# [Traceable]
# -----------------------------------------------------------------------------


class Traceable(ABC):
    """
    Abstract base class for traceable entities in zyx. Provides:
    1. Dynamic renaming with integrated tracing
    2. Hierarchical parent-child relationship support
    3. Context-aware tracing with proper module scoping
    4. Metadata storage for additional tracing context

    Subclasses must implement:
    - _get_trace_module(): Returns the module type for tracing
    - visualize(): Custom visualization logic
    """

    def __init__(self, name: str, parent: Optional[Traceable] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new traceable entity.

        Args:
            name (str): Unique identifier/name for this entity
            parent (Optional[Traceable]): Parent entity for hierarchical tracing
            metadata (Optional[Dict[str, Any]]): Additional context for tracing
        """
        self._name = name
        self._original_name = name
        self._parent = parent
        self._metadata = metadata or {}

        try:
            # Register with tracer system
            _tracer.register_entity(self._original_name, self._name)
            # Log creation in proper module context
            with self.trace_context():
                _tracer.log(
                    "create",
                    {
                        "name": self._name,
                        "parent": self._parent.name if self._parent else None,
                        "metadata": self._metadata,
                    },
                )
        except Exception as e:
            logger.error(f"Failed to initialize Traceable entity '{name}': {e}")
            raise

    @property
    def name(self) -> str:
        """Current name of the entity."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        """
        Update entity name with proper tracing.

        Args:
            new_name (str): New name for the entity
        """
        try:
            old_name = self._name
            self._name = new_name

            # Update tracer registration
            _tracer.register_entity(self._original_name, new_name)

            # Log the rename event
            with self.trace_context():
                _tracer.log("rename", {"old_name": old_name, "new_name": new_name})
        except Exception as e:
            logger.error(f"Failed to rename entity from '{old_name}' to '{new_name}': {e}")
            self._name = old_name  # Rollback on error
            raise

    @property
    def parent(self) -> Optional[Traceable]:
        """Parent entity in tracing hierarchy."""
        return self._parent

    @parent.setter
    def parent(self, new_parent: Optional[Traceable]):
        """
        Update parent entity with tracing.

        Args:
            new_parent (Optional[Traceable]): New parent entity
        """
        try:
            old_parent = self._parent
            self._parent = new_parent

            # Log parent change
            with self.trace_context():
                _tracer.log(
                    "parent_change",
                    {
                        "old_parent": old_parent.name if old_parent else None,
                        "new_parent": new_parent.name if new_parent else None,
                    },
                )
        except Exception as e:
            logger.error(f"Failed to update parent for entity '{self.name}': {e}")
            self._parent = old_parent  # Rollback on error
            raise

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata dictionary for additional tracing context."""
        return self._metadata

    def add_metadata(self, key: str, value: Any):
        """
        Add metadata entry with tracing.

        Args:
            key (str): Metadata key
            value (Any): Metadata value
        """
        try:
            self._metadata[key] = value
            with self.trace_context():
                _tracer.log("metadata_add", {"key": key, "value": value})
        except Exception as e:
            logger.error(f"Failed to add metadata '{key}' for entity '{self.name}': {e}")
            raise

    @abstractmethod
    def _get_trace_module(self) -> str:
        """
        Get the module type for this traceable entity.
        Must be implemented by subclasses.

        Returns:
            str: Module type (e.g., 'agent', 'graph', etc.)
        """
        pass

    @contextmanager
    def trace_context(self) -> Generator[TraceContext, None, None]:
        """
        Context manager for tracing operations on this entity.
        Ensures proper module context and hierarchical event propagation.

        Yields:
            TraceContext: Context object with tracing information
        """
        try:
            # Create context info using the imported TraceContext
            ctx = TraceContext(module=self._get_trace_module(), entity=self._original_name, active=True)

            # Enter tracer module context
            with _tracer.module_context(ctx.module, ctx.entity):
                yield ctx

        except Exception as e:
            logger.error(f"Error in trace context for entity '{self.name}': {e}")
            raise

    def log(self, event: str, data: Optional[Dict[str, Any]] = None):
        """
        Log a trace event in this entity's context.

        Args:
            event (str): Event to log (e.g., 'completion:request')
            data (Optional[Dict[str, Any]]): Additional event data
        """
        try:
            with self.trace_context():
                _tracer.log(event, data)
        except Exception as e:
            logger.error(f"Failed to log event '{event}' for entity '{self.name}': {e}")
            raise

    @abstractmethod
    def visualize(self):
        """
        Implement custom visualization logic.
        Must be implemented by subclasses.
        """
        pass

    def __repr__(self) -> str:
        """String representation including hierarchy info."""
        parent_str = f", parent='{self.parent.name}'" if self.parent else ""
        return f"<{self.__class__.__name__}(name='{self.name}'{parent_str})>"
