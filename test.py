from zyx.base_client import BaseClient as Completions
from pydantic import BaseModel

completions = Completions(verbose=True)

response = completions.chat_completion(
    "hello", model = "gpt-4o-mini"
)

class Response(BaseModel):
    message: str

response = completions.chat_completion(
    "hello", model = "gpt-4o-mini", response_model = Response
)

print(response)

response = completions.chat_completion(
    [
        [
            {"role": "user", "content": "hello"}
        ],
        [
            {"role": "user", "content": "hi"}
        ]
    ],
    model = "gpt-4o-mini",
)

print(response)

response = completions.chat_completion(
    [
        [
            {"role": "user", "content": "hello"}
        ],
        [
            {"role": "user", "content": "hi"}
        ]
    ],
    model = "openai/gpt-4o-mini",
    response_model = Response
)

print(response)