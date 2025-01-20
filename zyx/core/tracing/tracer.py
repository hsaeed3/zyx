"""
### zyx.core.tracing.tracer

Core tracing module: contains the `Tracer` & a global `tracer` singleton.

Example:
    ```python
    from zyx.core.tracing.tracer import tracer

    # Enable tracing for specific patterns
    tracer.trace(["agent(*):move", "graph(*):add_node"])

    # Register a receiver
    tracer.register_receiver("console", ConsoleReceiver())
    tracer.enable_receiver("console")

    # Add an event listener
    def on_move(event):
        print(f"Agent moved to {event.data['position']}")
    tracer.on("agent(*):move", on_move)

    # Emit an event
    tracer.emit("agent(bob):move", {"position": (1, 2)})
    ```

Args:
    None

Returns:
    None
"""
from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

from .types import (
    TracingEvent,
    TracingListener,
    TracingPattern,
    TracingReceiver,
    TracingRegistry,
)

logger = logging.getLogger(__name__)


class Tracer:
    """
    Tracer class responsible for:
      - Managing tracing patterns and modules
      - Handling event listeners
      - Emitting events to receivers

    Args:
        None

    Returns:
        None

    Example:
        ```python
        tracer = Tracer()
        tracer.trace(["agent(*):move"])
        tracer.emit("agent(bob):move", {"position": (1,2)})
        ```
    """

    def __init__(self) -> None:
        """
        Initialize the tracer with no modules, patterns, or receivers.

        Args:
            None

        Returns:
            None
        """
        self.verbose_enabled: bool = False
        self.verbose_enabled_all: bool = False

        self.modules: Dict[str, TracingRegistry] = {}
        self.patterns: Set[TracingPattern] = set()
        self.listeners: Dict[str, TracingListener] = {}
        self.receivers: Dict[str, TracingReceiver] = {}
        self.active_receivers: Dict[str, TracingReceiver] = {}

        logger.debug("Core Tracer initialized.")

    # -------------------------------------------------------------------------
    # [Toggles]
    # -------------------------------------------------------------------------
    def enable_all(self) -> None:
        """
        Enable tracing for all modules (equivalent to tracing '*:all').

        Args:
            None

        Returns:
            None

        Example:
            ```python
            tracer.enable_all()  # Enables tracing for everything
            ```
        """
        self.patterns.clear()
        for module in self.modules:
            self.patterns.add(TracingPattern.parse(f"{module}:all"))
        self.verbose_enabled = True
        self.verbose_enabled_all = True
        logger.debug("Enabled all tracing.")

    def disable_all(self) -> None:
        """
        Disable tracing for all modules.

        Args:
            None

        Returns:
            None

        Example:
            ```python
            tracer.disable_all()  # Disables all tracing
            ```
        """
        self.patterns.clear()
        self.verbose_enabled = False
        self.verbose_enabled_all = False
        logger.debug("Disabled all tracing.")

    def enable_receiver(self, name: str) -> None:
        """
        Enable a previously registered receiver by name.

        Args:
            name (str): Name of the receiver to enable

        Returns:
            None

        Raises:
            ValueError: If receiver is not registered

        Example:
            ```python
            tracer.enable_receiver("console")
            ```
        """
        if name not in self.receivers:
            raise ValueError(f"Receiver not registered: {name}")

        receiver = self.receivers[name]
        self.active_receivers[name] = receiver
        receiver.start()

        # Add all current listeners to the newly started receiver
        for listener in self.listeners.values():
            receiver.add_listener(listener)

        logger.debug(f"Enabled receiver: {name}")

    def disable_receiver(self, name: str) -> None:
        """
        Disable a currently active receiver by name.

        Args:
            name (str): Name of the receiver to disable

        Returns:
            None

        Example:
            ```python
            tracer.disable_receiver("console")
            ```
        """
        if name not in self.active_receivers:
            logger.warning(f"Receiver not active or does not exist: {name}")
            return

        del self.active_receivers[name]
        logger.debug(f"Disabled receiver: {name}")

    # -------------------------------------------------------------------------
    # [Modules & Registration]
    # -------------------------------------------------------------------------
    def register_module(self, name: str) -> None:
        """
        Register a new module with the tracer.

        Args:
            name (str): Name of the module to register

        Returns:
            None

        Example:
            ```python
            tracer.register_module("agent")
            ```
        """
        if name in self.modules:
            logger.warning(f"Module '{name}' is already registered.")
            return
        self.modules[name] = TracingRegistry()
        logger.debug(f"Module '{name}' registered with Tracer.")

    def register_entity(self, module: str, entity: str) -> None:
        """
        Register a new entity within a module for individual tracing.

        Args:
            module (str): Name of the module containing the entity
            entity (str): Name/ID of the entity to register

        Returns:
            None

        Raises:
            ValueError: If module is not registered

        Example:
            ```python
            tracer.register_entity("agent", "bob")
            ```
        """
        if module not in self.modules:
            raise ValueError(f"Module '{module}' not registered.")

        pattern = f"{module}({entity})"
        self.modules[module].listeners[pattern] = TracingListener(
            pattern=pattern
        )
        logger.debug(f"Entity '{entity}' registered under module '{module}'.")

    def register_receiver(self, name: str, receiver: TracingReceiver) -> None:
        """
        Register a new receiver by name.

        Args:
            name (str): Name to register the receiver under
            receiver (TracingReceiver): Receiver instance to register

        Returns:
            None

        Example:
            ```python
            console = ConsoleReceiver()
            tracer.register_receiver("console", console)
            ```
        """
        if name in self.receivers:
            logger.warning(f"Receiver '{name}' is already registered.")
            return

        self.receivers[name] = receiver
        logger.debug(f"Receiver '{name}' registered.")

    # -------------------------------------------------------------------------
    # [Helpers]
    # -------------------------------------------------------------------------
    def is_pattern_enabled(self, pattern: Union[str, TracingPattern]) -> bool:
        """
        Check if a pattern (str or TracingPattern) is currently enabled.

        Args:
            pattern (Union[str, TracingPattern]): Pattern to check

        Returns:
            bool: True if pattern is enabled, False otherwise

        Example:
            ```python
            if tracer.is_pattern_enabled("agent(*):move"):
                print("Agent movement tracing is enabled")
            ```
        """
        if not isinstance(pattern, TracingPattern):
            pattern = TracingPattern.parse(pattern)

        if self.verbose_enabled_all:
            return True

        return any(p.matches(pattern) for p in self.patterns)

    # -------------------------------------------------------------------------
    # [Tracing]
    # -------------------------------------------------------------------------
    def trace(self, patterns: Union[str, List[str], bool]) -> None:
        """
        Enable or disable tracing for patterns.

        Args:
            patterns (Union[str, List[str], bool]): 
                - If bool: True enables all, False disables all
                - If str "all": enables all
                - If str: pattern to enable
                - If List[str]: patterns to enable

        Returns:
            None

        Example:
            ```python
            tracer.trace(["agent(*):move", "graph(*):add_node"])
            tracer.trace("all")  # Enable all
            tracer.trace(False)  # Disable all
            ```
        """
        logger.debug(f"Trace request: {patterns}")

        # Boolean case
        if isinstance(patterns, bool):
            if patterns:
                self.enable_all()
            else:
                self.disable_all()
            return

        # Single string case
        if isinstance(patterns, str):
            if patterns == "all":
                self.enable_all()
                return
            patterns = [patterns]

        # List of patterns
        for pattern in patterns:
            self.patterns.add(TracingPattern.parse(pattern))

        self.verbose_enabled = True
        self.verbose_enabled_all = False
        logger.debug(f"Enabled tracing for: {patterns}")

    # -------------------------------------------------------------------------
    # [Hooks]
    # -------------------------------------------------------------------------
    def emit(
        self,
        pattern: Union[str, TracingPattern],
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit (fire) an event matching a pattern, with optional data.

        Args:
            pattern (Union[str, TracingPattern]): Pattern for the event
            data (Optional[Dict[str, Any]]): Event data dictionary

        Returns:
            None

        Example:
            ```python
            tracer.emit("agent(bob):move", {"position": (1, 2)})
            ```
        """
        if not isinstance(pattern, TracingPattern):
            pattern = TracingPattern.parse(pattern)

        if not self.is_pattern_enabled(pattern):
            return

        event = TracingEvent(
            is_enabled=True,
            pattern=str(pattern),
            data=data or {}
        )

        # Send to active receivers
        for receiver in self.active_receivers.values():
            receiver.update(event)

        # Forward to local listeners
        self._forward_to_listeners(event)
        logger.debug(f"Emitted event: {pattern}")

    def _forward_to_listeners(self, event: TracingEvent) -> None:
        """
        Forward an event to any local listeners that match the pattern.
        Remove 'once' listeners if they've been called already.

        Args:
            event (TracingEvent): Event to forward to listeners

        Returns:
            None
        """
        pattern = TracingPattern.parse(event.pattern)
        to_remove = []

        for key, listener in self.listeners.items():
            listener_pat = TracingPattern.parse(listener.pattern)
            if listener_pat.matches(pattern):
                try:
                    listener.callback(event)
                    if listener.once:
                        to_remove.append(key)
                except Exception as exc:
                    logger.error(f"Error in listener callback: {exc}")

        # Remove 'once' listeners
        for key in to_remove:
            del self.listeners[key]
            for receiver in self.active_receivers.values():
                receiver.remove_listener(key)

    def on(
        self,
        pattern: Union[str, TracingPattern],
        callback: Callable[[TracingEvent], None],
        once: bool = False
    ) -> None:
        """
        Register a new local event listener for the specified pattern.

        Args:
            pattern (Union[str, TracingPattern]): Pattern to listen for
            callback (Callable[[TracingEvent], None]): Function to call on event
            once (bool): If True, remove listener after first event

        Returns:
            None

        Example:
            ```python
            def on_move(event):
                print(f"Agent moved to {event.data['position']}")
            tracer.on("agent(*):move", on_move)
            ```
        """
        if not isinstance(pattern, TracingPattern):
            pattern = TracingPattern.parse(pattern)

        listener_str = str(pattern)
        listener = TracingListener(
            pattern=listener_str,
            callback=callback,
            once=once
        )
        self.listeners[listener_str] = listener

        # Also add to all active receivers
        for receiver in self.active_receivers.values():
            receiver.add_listener(listener)

        logger.debug(f"Registered listener for pattern: {pattern}")

    def off(
        self,
        pattern: Optional[Union[str, TracingPattern]] = None
    ) -> None:
        """
        Remove event listener(s). If no pattern is provided, remove all.

        Args:
            pattern (Optional[Union[str, TracingPattern]]): Pattern to remove,
                or None to remove all listeners

        Returns:
            None

        Example:
            ```python
            tracer.off("agent(*):move")  # Remove specific listener
            tracer.off()  # Remove all listeners
            ```
        """
        if pattern is None:
            # Remove all
            self.listeners.clear()
            for receiver in self.active_receivers.values():
                for key in list(receiver.listeners.keys()):
                    receiver.remove_listener(key)
            logger.debug("Removed all listeners.")
            return

        if not isinstance(pattern, TracingPattern):
            pattern = TracingPattern.parse(pattern)

        to_remove = []
        for key, listener in self.listeners.items():
            if TracingPattern.parse(listener.pattern).matches(pattern):
                to_remove.append(key)

        for key in to_remove:
            del self.listeners[key]
            for receiver in self.active_receivers.values():
                receiver.remove_listener(key)

        logger.debug(f"Removed listeners matching pattern: {pattern}")


#: Global Tracer instance for convenience.
tracer = Tracer()