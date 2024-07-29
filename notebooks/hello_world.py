from zyx.ai._completion import completion
from pydantic import BaseModel
class ResponseModel(BaseModel):
    urls : list[str]
    winner : str
    score : str
    
completion("who won the 2024 euro cup final?", tools = ["web"], response_model = ResponseModel,
               model = "openai/gpt-3.5-turbo", verbose = True)