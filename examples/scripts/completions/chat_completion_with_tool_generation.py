# Chat Completion with Tool Generation

import zyx as z

# defining tools as strings generates the code for the tools
#   using zyx.coder() ; this code is then executed

# lets run a simple example
response = z.completion(
    "What os am i running?",
    tools = ["run_cli_command"]
)


print(response.choices[0].message.content)