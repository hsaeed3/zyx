import zyx

class ExtractionModel(zyx.BaseModel):
    name : str
    age : int
    
text = "There once was a strange man. he really was odd. the man was 30. he was known to all as john"

print(zyx.extract(ExtractionModel, text, model = "ollama/llama3.1"))
print(zyx.generate(ExtractionModel))