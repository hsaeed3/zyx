from zyx.core.utils import convert_to_openai_tool

from pydantic import BaseModel, Field

class ExampleModel(BaseModel):
    """An example model."""
    name: str = Field(..., description="The name of the item")
    count: int = Field(..., description="The count of items")

def example_function(x: int, y: str) -> str:
    """An example function.
    
    Args:
        x: An integer parameter
        y: A string parameter
    
    Returns:
        A string result
    """
    pass

# Convert Pydantic model
pydantic_result = convert_to_openai_tool(ExampleModel)
print("Pydantic result:", pydantic_result)

# Convert Python function
function_result = convert_to_openai_tool(example_function)
print("Function result:", function_result)