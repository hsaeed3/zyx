from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Type, Literal
from . import client as clienttypes

class Document(BaseModel):
    content: Any
    metadata: Dict[str, Any]
    messages: Optional[List[Dict[str, Any]]] = []

    class Config:
        arbitrary_types_allowed = True

    def setup_messages(self):
        self.messages = [
            {
                "role": "system",
                "content": """
You are a world class document understanding assistant. You are able to 
understand the content of a document and answer questions about it.
"""
            },
            {
                "role": "user",
                "content": "What is the document?"
            },
            {
                "role": "assistant",
                "content": f"""
Here's a full overview of the document! \n
Document Metadata: {self.metadata} \n\n
Document Content: {self.content}
"""
            }
        ]

    def query(
            self,
            prompt: str,
            model: str = "gpt-4o-mini",
            client: Literal["openai", "litellm"] = "openai",
            response_model: Optional[Type[BaseModel]] = None,
            mode: Optional[clienttypes.InstructorMode] = "tool_call",
            max_retries: Optional[int] = 3,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            organization: Optional[str] = None,
            run_tools: Optional[bool] = True,
            tools: Optional[List[clienttypes.ToolType]] = None,
            parallel_tool_calls: Optional[bool] = False,
            tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[List[str]] = None,
            stream: Optional[bool] = False,
            verbose: Optional[bool] = False
    ):
        from ..completions.client import completion

        if not self.messages:
            self.setup_messages()

        self.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        if response_model or tools:
            print("ResponseModel & Tools not supported yet for Document.query()")

        response = completion(
            messages=self.messages,
            model=model,
            client=client,
            mode=mode,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            verbose=verbose
        )

        if response:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }
            )

        return response