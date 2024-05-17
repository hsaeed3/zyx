import zyx

response = zyx.completion("How are you today?")
print(response)

instruct = zyx.Instructor("A fastapi expert")
response = instruct.instruct("Write me a python script creating an inference endpoint")
print(response)

class Response(instruct.BaseModel):
    explanation : str
    code : str

response = instruct.instruct("Write me a python script creating an inference endpoint", response_model=Response)
print(response)