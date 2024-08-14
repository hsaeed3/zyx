_all_ = ["cli"]

# --- zyx ----------------------------------------------------------------

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog
from textual.containers import VerticalScroll
from typing import Union, Optional, List, Callable
from ..types import ClientModeParams
from ..core.ext import BaseModel

class ChatApp(App):

    def __init__(
        self,
        messages: Union[str, list[dict]] = None,
        model: Optional[str] = "gpt-4o-mini",
        tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
        run_tools: Optional[bool] = True,
        response_model: Optional[BaseModel] = None,
        mode: Optional[ClientModeParams] = "tools",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = 3,
        verbose: Optional[bool] = False,
        **kwargs
    ):
        from .main import Client

        super().__init__()
        self.client = Client()
        self.chat_history = self.client.format_messages(messages) if messages else []
        self.model = model
        self.tools = tools
        self.run_tools = run_tools
        self.response_model = response_model
        self.mode = mode
        self.base_url = base_url
        self.api_key = api_key
        self.organization = organization
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.verbose = verbose
        self.kwargs = kwargs

    def compose(self) -> ComposeResult:
        # Define the layout of the chat app
        with VerticalScroll():
            yield RichLog(id="chat_display")
            yield Input(placeholder="Type your message...", id="input_field")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_message = event.value.strip()
        if user_message:
            self.chat_history.append({"role": "user", "content": user_message})
            self.query_one("#chat_display", RichLog).write(f"[bold blue]User:[/bold blue] {user_message}\n", markup=True)

            response = self.client.completion(
                messages=self.chat_history,
                model=self.model,
                tools=self.tools,
                run_tools=self.run_tools,
                response_model=self.response_model,
                mode=self.mode,
                base_url=self.base_url,
                api_key=self.api_key,
                organization=self.organization,
                top_p=self.top_p,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=self.max_retries,
                verbose=self.verbose,
                **self.kwargs
            )

            assistant_reply = response.choices[0].message['content']
            self.chat_history.append({"role": "assistant", "content": assistant_reply})

            self.query_one("#chat_display", RichLog).write(f"[bold green]Assistant:[/bold green] {assistant_reply}\n", markup=True)

            self.query_one("#input_field", Input).value = ""

def cli(
    messages: Union[str, list[dict]] = None,
    model: Optional[str] = "gpt-4o-mini",
    tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
    run_tools: Optional[bool] = True,
    response_model: Optional[BaseModel] = None,
    mode: Optional[ClientModeParams] = "tools",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    verbose: Optional[bool] = False,
    **kwargs
):
    ChatApp(
        messages=messages,
        model=model,
        tools=tools,
        run_tools=run_tools,
        response_model=response_model,
        mode=mode,
        base_url=base_url,
        api_key=api_key,
        organization=organization,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose,
        **kwargs
    ).run()