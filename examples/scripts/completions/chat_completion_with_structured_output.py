# Chat Completion with Structured Outputs

import zyx as z
from pydantic import BaseModel

# lets create a simple model
class User(BaseModel):
    name : str
    age : int

response = z.completion(
    "Extract John is 20 years old",
    response_model = User
)

print("Structured Response: \n\n")
print(response)
print("\n\n")

# response_model can also be a simple type
response = z.completion(
    "Extract age for : John is 20 years old",
    response_model = int
)

print("Integer Response: \n\n")
print(response)
print("\n\n")

# lets do another example
response = z.completion(
    "Extract name for : John is 20 years old",
    response_model = str
)

print("String Response: \n\n")
print(response)
print("\n\n")