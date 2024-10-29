from ...resources.types import completion_create_params as params

from pydantic import BaseModel, Field
from typing import Callable, Optional, Literal, get_type_hints, Any, Union, Type
import traceback
from rich.progress import Progress, SpinnerColumn, TextColumn
from ...resources.utils import repl
import re

Client = Type["Client"]

class FunctionResponse(BaseModel):
    code: str
    output: Any


def prompt_user_library_install(libs: str) -> None:
    """Prompts user to install the required libraries for the function to run,
    installs if user enters y"""
    import subprocess
    import sys

    print(f"The function requires the following libraries to run: {libs}")
    install_prompt = input("Do you want to install these libraries? (y/n): ")

    if install_prompt.lower() == "y":
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs.split(",")]
        )
        print("Libraries installed successfully.")


def function(
    model: Union[str, params.ChatModel] = "gpt-4o-mini",
    provider: Literal["litellm", "openai"] = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: params.InstructorMode = "tool_call",
    mock: bool = False,
    return_code: bool = False,
    progress_bar: Optional[bool] = True,
    client : Optional[Client] = None,
    verbose: bool = False,
    **kwargs,
) -> Callable[[Callable], Callable]:
    """A quick abstraction to create both mock & generated runnable python functions, using LLMs.

    Parameters:
        model (str): The model to use for the function.
        api_key (Optional[str]): The API key to use for the function.
        base_url (Optional[str]): The base URL to use for the function.
        mode (Literal["json", "md_json", "tools"]): The mode to use for the function.
        mock (bool): Whether to use the mock mode for the function.
        kwargs (dict): Additional keyword arguments to pass to the function.

    Returns:
        The function response or the generated code response (Any).
    """

    from ..base_client import Client

    def decorator(f: Callable) -> Callable:
        from pydantic import create_model
        from functools import wraps
        import tenacity

        @wraps(f)
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(3),
            retry=tenacity.retry_if_exception_type((RuntimeError, Exception)),
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        )
        def wrapper(*args, **kwargs):
            type_hints = get_type_hints(f)
            return_type = type_hints.pop("return", Any)
            function_args = {k: v for k, v in type_hints.items()}
            input_dict = dict(zip(function_args.keys(), args))
            input_dict.update(kwargs)

            if not mock:

                class CodeGenerationModel(BaseModel):
                    code: str = Field(
                        ..., description="Complete Python code as a single string"
                    )
                    explanation: Optional[str] = Field(
                        None,
                        description="An optional explanation for the code. Not required, but any comments should go here.",
                    )

                error_context = ""
                try:
                    # Prepare the import statement and return type
                    import_statement = "import typing\n"
                    return_type_str = (
                        f"typing.Any" if return_type == Any else str(return_type)
                    )

                    # First, generate the function definition
                    base_instructions = """
                    You are a Python code generator. Your goal is to generate Python code based on the given description.

                    # Instructions
                    - Generate the complete Python code as a single string.
                    - The code should define a single function with all necessary imports inside it.
                    - All arguments in the code should be typed.
                    - The function must include a clear docstring.
                    - All imports must be nested inside the function body.
                    - The function should return the final result directly.
                    - Do not include any calls to the function.
                    """

                    system_message = base_instructions
                    user_message = f"""Generate a Python function that matches this signature:
                    Function: {f.__name__}
                    Arguments: {function_args}
                    Return type: {return_type_str}
                    Description: {f.__doc__}
                    """

                    completion_client = Client(
                        api_key=api_key,
                        base_url=base_url,
                        provider=provider,
                        verbose=verbose,
                    ) if client is None else client

                    if progress_bar:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            transient=True
                        ) as progress:
                            task_id = progress.add_task("Constructing...", total=None)

                            response = completion_client.completion(
                                messages=[
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": user_message},
                                ],
                                model=model,
                                response_model=CodeGenerationModel,
                                mode=mode,
                                progress_bar=False,
                                **kwargs,
                            )

                            progress.update(task_id, completed=1)
                    else:
                        response = completion_client.completion(
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message},
                            ],
                            model=model,
                            response_model=CodeGenerationModel,
                            mode=mode,
                            progress_bar=False,
                            **kwargs,
                        )

                    if verbose:
                        print(f"Generated function:\n{response.code}")

                    # Get the function definition
                    function_code = response.code

                    # Now create the execution code by adding the function call
                    args_str = ", ".join([f"{k}={repr(v)}" for k, v in input_dict.items()])
                    execution_code = f"""
{function_code}

result = {f.__name__}({args_str})
"""

                    if return_code:
                        return FunctionResponse(code=function_code, output=None)

                    try:
                        # Execute the complete code
                        output = repl.execute_in_sandbox(
                            execution_code,
                            verbose=verbose,
                            return_result=True
                        )

                        if verbose:
                            print(f"Execution successful, result type: {type(output)}")

                        return output if not return_code else FunctionResponse(code=function_code, output=output)

                    except Exception as e:
                        last_error = str(e)
                        last_code = execution_code
                        raise RuntimeError(f"Error in code execution: {str(e)}")

                except ImportError as e:
                    print(f"Import error: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    prompt_user_library_install(str(e).split("'")[1])
                    raise RuntimeError(f"Import error: {str(e)}")

                except Exception as e:
                    print(f"Error in code generation or execution: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    raise RuntimeError(
                        f"Error in code generation or execution: {str(e)}"
                    )

            else:
                FunctionResponseModel = create_model(
                    "FunctionResponseModel",
                    output=(return_type, ...),
                )
                messages = [
                    {
                        "role": "system",
                        "content": f"""
                        You are a Python function emulator. Your goal is to simulate the response of this Python function:
                        Function: {f.__name__}
                        Arguments and their types: {function_args}
                        Return type: {return_type}
                        Description: {f.__doc__}
                        Respond only with the output the function would produce, without any additional explanation.
                        """,
                    },
                    {"role": "user", "content": f"Function inputs: {input_dict}"},
                ]

                completion_client = Client(
                    api_key=api_key, base_url=base_url, provider=provider, verbose=verbose
                ) if client is None else client

                if progress_bar:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True
                    ) as progress:
                        task_id = progress.add_task("Simulating Function...", total=None)

                        response = completion_client.completion(
                            messages=messages,
                            model=model,
                            response_model=FunctionResponseModel,
                            mode="markdown_json_mode"
                            if model.startswith(("ollama/", "ollama_chat/"))
                            else mode,
                            progress_bar=False,
                            **kwargs,
                        )

                        progress.update(task_id, completed=1)
                else:
                    response = completion_client.completion(
                        messages=messages,
                        model=model,
                        response_model=FunctionResponseModel,
                        mode="markdown_json_mode"
                        if model.startswith(("ollama/", "ollama_chat/"))
                        else mode,
                        progress_bar=False,
                        **kwargs,
                    )

                if return_code:
                    return FunctionResponse(code="", output=response.output)
                return response.output

        return wrapper

    return decorator


if __name__ == "__main__":

    @function(mock=False, verbose=True)
    def add(a: int, b: int) -> int:
        """
        Add two numbers.
        """

    print(add(1, 2))

    @function(mock=False, verbose=False)
    def get_logger(name: str):
        """
        Get a logger with the given name, configured with a StreamHandler and INFO level.
        """

    logger = get_logger("my_logger")
    logger.info("Hello, world!")