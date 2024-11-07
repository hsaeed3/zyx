from openai import pydantic_function_tool
from pydantic import BaseModel
from typing import Literal
from rich import print

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

print(
    pydantic_function_tool(Sentiment)
)

