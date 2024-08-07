from typing import (
    Annotated,
    Optional,
    Sequence,
    Union,
    Literal,
    Any,
    TypedDict,
)
import json
from pydantic import BaseModel, Field
from rich import print
import operator

# ==============================================================================


class zyxLanguageClientParams(BaseModel):
    model: Optional[str] = "openai/gpt-4o-mini"
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0


class zyxLanguageGraphClient:
    from langchain_core.tools import BaseTool

    def __init__(
        self,
        model: Optional[str] = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 0,
        verbose: Optional[bool] = False,
    ):
        self.verbose = verbose
        self._connect_llm(
            model, api_key, base_url, organization, max_tokens, temperature
        )
        if self.verbose:
            print(
                f"Connected to [bold green]{self.params.model}[/bold green] using [bold dark_green]{self.params.provider}[/bold dark_green] client provider"
            )

    def _connect_llm(
        self,
        model: str,
        api_key: str = None,
        base_url: str = None,
        organization: str = None,
        max_tokens: int = None,
        temperature: float = 0,
    ):
        if model.startswith("openai/") or model.startswith("gpt-"):
            try:
                if model.startswith("openai/"):
                    model = model.replace("openai/", "")
                self.llm = self._connect_openai_llm(
                    model, api_key, base_url, max_tokens, temperature
                )
            except Exception as e:
                raise e
            self.params = zyxLanguageClientParams(
                model=model,
                provider="OpenAI",
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif model.startswith("ollama/"):
            model = model.replace("ollama/", "")
            try:
                self.llm = self._connect_ollama_llm(
                    model=model, temperature=temperature
                )
            except Exception as e:
                raise e
            self.params = zyxLanguageClientParams(
                model=model,
                provider="Ollama",
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            try:
                self.llm = self._connect_litellm_llm(
                    model, base_url, organization, max_tokens, temperature
                )
            except Exception as e:
                raise e
            self.params = zyxLanguageClientParams(
                model=model,
                provider="LiteLLM",
                organization=organization,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    def _connect_litellm_llm(
        self,
        model: str,
        base_url: str = None,
        organization: str = None,
        max_tokens: int = None,
        temperature: float = 0,
    ):
        from langchain_community.chat_models.litellm import ChatLiteLLM

        return ChatLiteLLM(
            model=model,
            api_base=base_url,
            organization=organization,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=self.verbose,
        )

    def _connect_openai_llm(
        self,
        model: str,
        api_key: str = None,
        base_url: str = None,
        max_tokens: int = None,
        temperature: float = 0,
    ):
        from langchain_openai.chat_models.base import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=self.verbose,
        )

    def _connect_ollama_llm(self, model: str, temperature: float = 0):
        from langchain_ollama.chat_models import ChatOllama

        return ChatOllama(model=model, temperature=temperature, verbose=self.verbose)

    def _setup_graph_state(self):
        from langchain_core.messages.base import BaseMessage

        class zyxAgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]

        self.state = zyxAgentState
        if self.verbose:
            print("Initialized [bold deep_sky_blue4]Agent State[/bold deep_sky_blue4].")

    @staticmethod
    def _build_messages(messages: Union[str, list[dict]]):
        from langchain_core.messages.human import HumanMessage
        from langchain_community.adapters.openai import convert_openai_messages

        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        elif isinstance(messages, list):
            messages = convert_openai_messages(messages)
        return messages

    def _bind_tools(
        self,
        tools: Optional[list[BaseTool]],
        response_model: Optional[BaseModel] = None,
    ):
        if not tools:
            self.llm = self.llm.bind_tools([response_model])
            if self.verbose:
                print(
                    f"Bound [bold green]{response_model.__name__}[/bold green] to the model."
                )
        elif not response_model:
            self.llm = self.llm.bind_tools(tools, tool_choice="any")
            if self.verbose:
                print(
                    f"Bound [bold green]{len(tools)} tools[/bold green] to the model."
                )
        else:
            self.llm = self.llm.bind_tools(tools + [response_model], tool_choice="any")
            if self.verbose:
                print(
                    f"Bound [bold green]{len(tools)} tools[/bold green] and [bold blue]{response_model.__name__}[/bold blue] to the model."
                )

    def _build_graph(
        self,
        tools: Optional[list[BaseTool]],
        response_model: Optional[BaseModel] = None,
    ):
        from langgraph.graph.state import StateGraph, START, END
        from langgraph.prebuilt.tool_node import ToolNode

        self._bind_tools(tools=tools, response_model=response_model)
        tool_node = ToolNode(tools if tools else [response_model])

        def call_model(state: self.state):
            messages = state["messages"]
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        def route(state) -> Literal["action", END]:
            messages = state["messages"]
            last_message = messages[-1]
            if not last_message.additional_kwargs.get("tool_calls"):
                return END
            if (
                response_model
                and last_message.additional_kwargs["tool_calls"][0]["function"]["name"]
                == response_model.__name__
            ):
                return END
            return "action"

        self.graph = StateGraph(self.state)
        self.graph.add_node("agent", call_model)
        self.graph.add_node("action", tool_node)
        self.graph.add_edge(START, "agent")
        self.graph.add_conditional_edges("agent", route)
        self.graph.add_edge("action", "agent")

        self.app = self.graph.compile()

    def _handle_completion(
        self,
        messages: Union[str, list[dict]] = None,
        tools: Optional[list[BaseTool]] = None,
        response_model: Optional[BaseModel] = None,
    ):
        self._setup_graph_state()

        from langchain_core.messages.ai import AIMessage

        class CompletionModel(BaseModel):
            response: str = Field(description="The complete response from the model")

        messages = self._build_messages(messages)
        if self.verbose:
            print(f"Formatted {len(messages)} messages.")

        if not response_model and not tools:
            return self.llm.invoke(messages)

        if not response_model:
            internal_response_model = CompletionModel
        else:
            internal_response_model = response_model
        self._build_graph(tools, internal_response_model)
        if self.verbose:
            print("Initialized Graph Workflow.")

        def parse_output(message: AIMessage) -> Any:
            if self.verbose:
                print(f"Parsing message: {message}")

            if message.additional_kwargs.get("tool_calls"):
                tool_call = message.additional_kwargs["tool_calls"][0]
                if tool_call["function"]["name"] == internal_response_model.__name__:
                    args = json.loads(tool_call["function"]["arguments"])
                    completion_result = internal_response_model(**args)
                    return completion_result
            return message.content

        inputs = {"messages": messages}
        output = self.app.invoke(inputs)
        if self.verbose:
            print(f"Graph output: {output}")
        return parse_output(output["messages"][-1])


# ==============================================================================


def _completion(
    messages: Union[str, list[dict]],
    model: Optional[str] = "openai/gpt-4o-mini",
    tools: Optional[list] = None,
    response_model: Optional[BaseModel] = None,
    verbose: Optional[bool] = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = 0,
):
    """"""
    client = zyxLanguageGraphClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
    )
    return client._handle_completion(messages, tools, response_model)


if __name__ == "__main__":

    class ResponseClassModel(BaseModel):
        feelings: list
        emotions: list

    print(
        _completion(
            "Hi how are yaaa",
            model="ollama_chat/llama3.1",
            response_model=ResponseClassModel,
        )
    )
