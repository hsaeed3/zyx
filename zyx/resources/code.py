"""zyx.resources.code

Provides a Code resource that uses Monty to safely execute Python code written
by AI agents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, TypeAlias, TYPE_CHECKING

from pydantic_ai.toolsets import FunctionToolset

from .abstract import AbstractResource

if TYPE_CHECKING:
    try:
        from pydantic_monty import Monty  # type: ignore[import-untyped]
    except ImportError:
        Monty: TypeAlias = Any


def _get_monty():
    try:
        import pydantic_monty  # type: ignore[import-untyped]

        return pydantic_monty
    except ImportError:
        raise ImportError(
            "To use the `Code` resource, you must first install the `pydantic-monty` library.\n"
            "You can install it by using one of the following commands:\n"
            "```bash\n"
            "pip install zyx[code]\n"
            "pip install pydantic-monty\n"
            "```"
        )


@dataclass(init=False)
class Code(AbstractResource):
    """
    A resource that allows safely executing Python code using Monty, a minimal,
    secure Python interpreter written in Rust.

    This resource enables AI agents to write and execute Python code without
    the overhead of container-based sandboxes, while maintaining strict security
    controls over filesystem, network, and environment access.
    """

    script_name: str = field(default="code.py")
    """The name of the script file for error reporting and type checking."""

    type_check: bool = field(default=True)
    """Whether to enable type checking for the code."""

    type_check_stubs: str | None = field(default=None)
    """Optional type definitions/stubs for type checking."""

    external_functions: Dict[str, Callable] = field(default_factory=dict)
    """Dictionary mapping function names to their implementations that can be
    called from the executed code."""

    def __init__(
        self,
        *,
        name: str = "code",
        writeable: bool = True,
        confirm: bool = False,
        script_name: str = "code.py",
        type_check: bool = True,
        type_check_stubs: str | None = None,
        external_functions: Dict[str, Callable] | None = None,
    ) -> None:
        super().__init__(name=name, writeable=writeable, confirm=confirm)
        self.script_name = script_name
        self.type_check = type_check
        self.type_check_stubs = type_check_stubs
        self.external_functions = external_functions or {}

    def get_description(self) -> str:
        return (
            f"Code execution environment using Monty. "
            f"Supports executing Python code safely with controlled access to "
            f"external functions: {', '.join(self.external_functions.keys()) or 'none'}."
        )

    def get_state_description(self) -> str:
        return (
            f"Code resource ready for execution. "
            f"External functions available: {', '.join(self.external_functions.keys()) or 'none'}."
        )

    def _create_monty(
        self,
        code: str,
        inputs: List[str] | None = None,
    ) -> Monty:
        """Create a Monty instance for the given code."""
        return _get_monty().Monty(
            code,
            inputs=inputs or [],
            external_functions=list(self.external_functions.keys()),
            script_name=self.script_name,
            type_check=self.type_check,
            type_check_stubs=self.type_check_stubs,
        )

    async def _run_async(
        self,
        code: str,
        inputs: Dict[str, Any] | None = None,
    ) -> Any:
        """Execute code asynchronously and return the result."""
        m = self._create_monty(
            code, inputs=list(inputs.keys()) if inputs else None
        )
        return await _get_monty().run_monty_async(
            m,
            inputs=inputs or {},
            external_functions=self.external_functions,
        )

    def _run_sync(
        self,
        code: str,
        inputs: Dict[str, Any] | None = None,
    ) -> Any:
        """Execute code synchronously and return the result."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call synchronous 'run_code' inside an async context. "
                "Use 'await async_run_code' instead."
            )
        else:
            return asyncio.run(self._run_async(code, inputs))

    async def async_run_code(
        self,
        code: str,
        inputs: Dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute Python code asynchronously using Monty and return the result.

        Args:
            code: The Python code to execute.
            inputs: Optional dictionary of input variables to pass to the code.

        Returns:
            The result of executing the code.
        """
        return await self._run_async(code, inputs)

    def run_code(
        self,
        code: str,
        inputs: Dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute Python code synchronously using Monty and return the result.

        Args:
            code: The Python code to execute.
            inputs: Optional dictionary of input variables to pass to the code.

        Returns:
            The result of executing the code.
        """
        return self._run_sync(code, inputs)

    def get_toolset(self) -> FunctionToolset:
        """Get the toolset for interacting with this code resource."""
        toolset = FunctionToolset()

        @toolset.tool
        def execute_code(
            code: str,
            inputs: Dict[str, Any] | None = None,
        ) -> Any:
            """
            Execute Python code safely using Monty.

            The code can call external functions that have been registered with
            this Code resource. The code runs in a secure sandbox with no access
            to filesystem, network, or environment variables unless explicitly
            provided via external functions.

            Args:
                code: The Python code to execute.
                inputs: Optional dictionary of input variables to pass to the code.

            Returns:
                The result of executing the code.
            """
            return self.run_code(code, inputs)

        return toolset
