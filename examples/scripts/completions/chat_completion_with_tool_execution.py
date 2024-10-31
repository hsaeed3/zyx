# Chat Completion with Automatic Tool Execution

# lets create a tool
import math

# you can use any function as a tool
# no need to decorate with @tool
# no need to add docstring if the tool is simple
def square_root(x : int) -> int:
    return math.sqrt(x)

# now we can use this tool in a chat completion
import zyx as z

response = z.completion(
    "What is the square root of 2300329944?",
    tools = [square_root]
)

print("Assistant Response: \n\n")
print(response.choices[0].message.content)
print("\n\n")

# setting run_tools to False will disable tool execution
response = z.completion(
    "What is the square root of 2300329944?",
    tools = [square_root],
    run_tools = False
)

# lets print the tool calls
print("Tool Calls: \n\n")
print(response.choices[0].message.tool_calls)
print("\n\n")