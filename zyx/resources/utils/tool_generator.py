from ..types import completion_create_params as params
import re


def _generate_tool(client, tool_name: str, model: str) -> params.Tool:
    """Dynamically generate a tool function from its name"""
    
    # Create a prompt that will generate a function based on the tool name
    function_prompt = f"""Create a Python function named '{tool_name}' that:
    1. Has typed arguments and return value
    2. Includes clear docstring
    3. Handles errors with try/except
    4. Places imports inside function body
    5. Implements core functionality implied by '{tool_name}'"""

    from ...functions.coder import coder

    # Use coder to generate the implementation
    generated_code = coder(
        description=function_prompt,
        model=model, 
        temperature=0.2,
        client=client,
        verbose=client.config.verbose,
        return_code=True,
        progress_bar=False
    )

    # Create function object from code
    namespace = {}
    try:
        exec(generated_code, namespace)
        generated_function = namespace[tool_name]
    except Exception as e:
        raise ValueError(f"Failed to create function object for {tool_name}: {str(e)}")

    # Create and return a Tool instance
    tool = params.Tool(
        name=tool_name,
        function=generated_function,
        description=generated_function.__doc__
    ) 

    return tool
