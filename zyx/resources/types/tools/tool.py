# xnano.shared.types.completions.tool
# tool calling types

from ...._exceptions import Yikes
from ...utils import function_calling
from pydantic import BaseModel, ConfigDict
from typing import Any, Callable, Type, Dict, Optional, Union


ToolType = Union[Callable, Type[BaseModel], Dict[str, Any]]


class Tool(BaseModel):
    """Internal tool class."""

    # arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # tool name (used for execution)
    name : Optional[str] = None

    # arguments (used for execution)
    arguments : Optional[Dict[str, Any]] = None

    # description
    description : Optional[str] = None

    # function
    function : Union[
        # openai function
        Dict[str, Any],
        # pydantic model
        Type[BaseModel],
        # callable
        Callable
    ]

    # formatted function (completion request)
    formatted_function : Optional[Dict[str, Any]] = None

    # execute helper (if callable)
    def _execute(self, verbose : bool = False, **kwargs) -> Any:
        """Executes the tool."""
        if isinstance(self.function, Dict):

            if verbose:
                Yikes(f"Tool is not callable: {self.function}")

            return None
        else:
            return self.function(**kwargs)
        

    def convert(self, verbose : bool = False) -> None:
        """Converts the tool to a formatted function."""
        self.formatted_function = function_calling.convert_to_openai_tool(self.function)

        self.name = function_calling.get_function_name(self.function)
        self.arguments = function_calling.get_function_arguments(self.function)
        self.description = function_calling.get_function_description(self.function)

        if verbose:
            print(f"Tool converted to: {self.formatted_function}")


