"""zyx.tools.tool"""

import asyncio
from typing import (
    Any,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

from pydantic_ai._run_context import get_current_run_context

from .._aliases import (
    _pydantic_ai_function_schema,
    _pydantic_ai_tools,
    PydanticAIRunContext,
)

__all__ = ("Tool",)


Deps = TypeVar("Deps")


ToolReturn = TypeVar("ToolReturn")


ToolParams = ParamSpec("ToolParams")


class Tool(_pydantic_ai_tools.Tool[Deps], Generic[Deps, ToolParams, ToolReturn]):
    """A tool that can be given to a model or an agent for tool calling and execution during
    a semantic operation.

    This is a wrapper around the `pydantic_ai.tools.Tool` class that allows for the use of
    the __call__ method to invoke the tool function directly.
    """

    def __init__(
        self,
        function: Callable[ToolParams, ToolReturn] | Callable[[PydanticAIRunContext[Deps], ToolParams], ToolReturn],
        *,
        takes_ctx: bool | None = None,
        max_retries: int | None = None,
        name: str | None = None,
        description: str | None = None,
        prepare: _pydantic_ai_tools.ToolPrepareFunc[_pydantic_ai_tools.ToolAgentDepsT] | None = None,
        docstring_format: _pydantic_ai_tools.DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[_pydantic_ai_tools.GenerateJsonSchema] = _pydantic_ai_tools.GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        function_schema: _pydantic_ai_function_schema.FunctionSchema | None = None,
    ) -> None:
        """
        Initialize a new `Tool` instance.

        Args:
            function: Callable[ToolParams, ToolReturn] | Callable[[PydanticAIRunContext[Deps], ToolParams], ToolReturn]
                The function to wrap as a `pydantic_ai.tools.Tool`.
            takes_ctx: bool | None = None
                Whether the tool takes a `PydanticAIRunContext` as its first argument.
            max_retries: int | None = None
                The maximum number of retries for the tool.
            name: str | None = None
                The name of the tool.
            description: str | None = None
                The description of the tool.
            prepare: _pydantic_ai_tools.ToolPrepareFunc[_pydantic_ai_tools.ToolAgentDepsT] | None = None
                The prepare function for the tool.
            docstring_format: _pydantic_ai_tools.DocstringFormat = 'auto'
                The format of the docstring for the tool.
            require_parameter_descriptions: bool = False
                Whether to require parameter descriptions for the tool.
            schema_generator: type[_pydantic_ai_tools.GenerateJsonSchema] = _pydantic_ai_tools.GenerateToolJsonSchema
                The schema generator for the tool.
            strict: bool | None = None
                Whether to enforce strict JSON schema validation for the tool.
            sequential: bool = False
                Whether the tool requires a sequential/serial execution environment.
            requires_approval: bool = False
                Whether the tool requires human-in-the-loop approval.
            metadata: dict[str, Any] | None = None
                The metadata for the tool.
            timeout: float | None = None
                The timeout for the tool.
            function_schema: _function_schema.FunctionSchema | None = None
                The `pydantic_ai._function_schema.FunctionSchema` for the tool.
        """
        super().__init__(
            function=function,
            takes_ctx=takes_ctx,
            max_retries=max_retries,
            name=name,
            description=description,
            prepare=cast(
                _pydantic_ai_tools.ToolPrepareFunc[object] | None, prepare
            ),
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
            sequential=sequential,
            requires_approval=requires_approval,
            metadata=metadata,
            timeout=timeout,
            function_schema=function_schema,
        )

    def _args_dict(self, *args: ToolParams.args, **kwargs: ToolParams.kwargs) -> dict[str, Any]:
        """Build and validate args dict from *args, **kwargs using the tool schema."""
        schema = self.function_schema
        if schema.positional_fields:
            args_dict = dict(zip(schema.positional_fields, args))
        else:
            args_dict = {}
        args_dict.update(kwargs)
        return schema.validator.validate_python(args_dict)

    def __call__(self, *args: ToolParams.args, **kwargs: ToolParams.kwargs) -> ToolReturn:
        """Invoke the tool synchronously. Uses pydantic_ai invocation when takes_ctx."""
        if self.takes_ctx:
            ctx = get_current_run_context()
            if ctx is None:
                raise RuntimeError(
                    "Tool requires a run context (takes_ctx=True). "
                    "Call from within an agent run, or use a tool that does not take RunContext."
                )
            args_dict = self._args_dict(*args, **kwargs)
            coro = self.function_schema.call(args_dict, ctx)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            raise RuntimeError(
                "Cannot call async tool synchronously from within an async context. "
                "Use await tool.acall(...) instead."
            )
        result = self.function(*args, **kwargs)
        if self.function_schema.is_async:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)
            raise RuntimeError(
                "Cannot call async tool synchronously from within an async context. "
                "Use await tool.acall(...) instead."
            )
        return result

    async def acall(self, *args: ToolParams.args, **kwargs: ToolParams.kwargs) -> ToolReturn:
        """Invoke the tool asynchronously. Uses pydantic_ai invocation when takes_ctx."""
        if self.takes_ctx:
            ctx = get_current_run_context()
            if ctx is None:
                raise RuntimeError(
                    "Tool requires a run context (takes_ctx=True). "
                    "Call from within an agent run, or use a tool that does not take RunContext."
                )
            args_dict = self._args_dict(*args, **kwargs)
            return await self.function_schema.call(args_dict, ctx)
        result = self.function(*args, **kwargs)
        if self.function_schema.is_async:
            return await result
        return result


def tool(
    function: Callable[ToolParams, ToolReturn] | Callable[[PydanticAIRunContext[Deps], ToolParams], ToolReturn],
    *,
    takes_ctx: bool | None = None,
    max_retries: int | None = None,
    name: str | None = None,
    description: str | None = None,
    prepare: _pydantic_ai_tools.ToolPrepareFunc[_pydantic_ai_tools.ToolAgentDepsT] | None = None,
    docstring_format: _pydantic_ai_tools.DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[_pydantic_ai_tools.GenerateJsonSchema] = _pydantic_ai_tools.GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    function_schema: _pydantic_ai_function_schema.FunctionSchema | None = None,
) -> Tool[Any, ToolParams, ToolReturn]:
    """Wrapper factory function for creating a `Tool` that can be passed to a model or an agent for tool calling and execution during a semantic operation."""
    return Tool(
        function=function,
        takes_ctx=takes_ctx,
        max_retries=max_retries,
        name=name,
        description=description,
        prepare=cast(
            _pydantic_ai_tools.ToolPrepareFunc[object] | None, prepare
        ),
        docstring_format=docstring_format,
        require_parameter_descriptions=require_parameter_descriptions,
        schema_generator=schema_generator,
        strict=strict,
        sequential=sequential,
        requires_approval=requires_approval,
        metadata=metadata,
        timeout=timeout,
        function_schema=function_schema,
    )