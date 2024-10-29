from ....client import Client, InstructorMode
from pydantic import BaseModel, Field
from typing import Callable, Optional, Literal, get_type_hints, Any
import traceback
import logging
import importlib
import subprocess
import sys
import os
import tempfile


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
    model: str = "gpt-4o-mini",
    client: Literal["litellm", "openai"] = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: InstructorMode = "tool_call",
    mock: bool = False,
    return_code: bool = False,
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

    def decorator(f: Callable) -> Callable:
        from pydantic import create_model
        from functools import wraps
        import tenacity
        import tempfile
        import sys
        import os
        import importlib

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

                    messages = [
                        {
                            "role": "system",
                            "content": f"""
                            ## CONTEXT ##
                            
                            You are a Python code generator. Your goal is to generate a Python function that matches this specification:
                            Function: {f.__name__}
                            Arguments and their types: {function_args}
                            Return type: {return_type_str}
                            Description: {f.__doc__} \n
                            
                            ## OBJECTIVE ##
                            
                            Generate the complete Python code as a single string, including all necessary import statements.
                            The code should define the function and include a call to the function with the provided inputs.
                            The last line should assign the result of the function call to a variable named 'result'.
                            Do not include any JSON encoding, printing, or explanations in your response.
                            Ensure all required modules are imported.
                            If your function returns an object, ensure it is properly configured and ready to be used.
                            {error_context}
                            """,
                        },
                        {
                            "role": "user",
                            "content": f"Generate code for the function with these inputs: {input_dict}",
                        },
                    ]

                    completion_client = Client(
                        api_key=api_key,
                        base_url=base_url,
                        provider=client,
                        verbose=verbose,
                    )

                    response = completion_client.completion(
                        messages=messages,
                        model=model,
                        response_model=CodeGenerationModel,
                        mode=mode,
                        **kwargs,
                    )

                    if verbose:
                        print(f"Code generation response: {response}")

                    # Prepend the import statement to the generated code
                    full_code = import_statement + response.code

                    if verbose:
                        print(f"Full generated code:\n{full_code}")

                    # Create a temporary Python file
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".py", delete=False
                    ) as temp_file:
                        temp_file.write(full_code)
                        temp_file_path = temp_file.name

                    # Add the current directory to sys.path to allow importing the temporary module
                    sys.path.insert(0, os.path.dirname(temp_file_path))

                    try:
                        # Execute the generated code in a local namespace
                        local_namespace = {}
                        exec_globals = globals().copy()

                        # Dynamically import required modules
                        for line in full_code.split("\n"):
                            if line.startswith("import ") or line.startswith("from "):
                                try:
                                    exec(line, exec_globals)
                                except ImportError as e:
                                    print(f"Failed to import: {line}. Error: {str(e)}")
                                    print(
                                        "Attempting to install the required package..."
                                    )
                                    package = line.split()[1].split(".")[0]
                                    subprocess.check_call(
                                        [
                                            sys.executable,
                                            "-m",
                                            "pip",
                                            "install",
                                            package,
                                        ]
                                    )
                                    exec(line, exec_globals)

                        exec(full_code, exec_globals, local_namespace)

                        if "result" not in local_namespace:
                            raise ValueError(
                                "No result object found in the generated code."
                            )

                        result = local_namespace["result"]

                        if verbose:
                            print(f"Result type: {type(result)}")
                            if isinstance(result, logging.Logger):
                                print(f"Logger name: {result.name}")
                                print(f"Logger level: {result.level}")
                                print(f"Logger handlers: {result.handlers}")

                        if return_code:
                            return FunctionResponse(code=full_code, output=result)
                        return result

                    finally:
                        # Clean up: remove the temporary file
                        os.unlink(temp_file_path)
                        sys.path.pop(0)

                except ImportError as e:
                    print(f"Import error: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    prompt_user_library_install(e.name)
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
                    api_key=api_key, base_url=base_url, provider=client, verbose=verbose
                )

                response = completion_client.completion(
                    messages=messages,
                    model=model,
                    response_model=FunctionResponseModel,
                    mode="markdown_json_mode"
                    if model.startswith(("ollama/", "ollama_chat/"))
                    else mode,
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
