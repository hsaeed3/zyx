__all__ = ["chat"]

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog
from textual.containers import VerticalScroll
from typing import Any, Union, Optional, List, Callable, Literal
from ...core.types import ClientModeParams
from ...core.main import BaseModel
from rich.text import Text


class ChatApp(App):
    def __init__(
        self,
        provider: Optional[Any] = None,
        messages: Union[str, list[dict]] = None,
        model: Optional[str] = "gpt-4o-mini",
        theme: Optional[Literal["light", "dark"]] = "dark",
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
        cutoff: Optional[int] = 95,
        **kwargs,
    ):
        try:
            from ..main import Client

            super().__init__()
            self.client = Client()
            self.provider = provider
            self.chat_history = (
                self.client.format_messages(messages) if messages else []
            )
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
            self.background = "#f0f0f0" if theme == "light" else "#1e1e1e"
            self.text = "#000000" if theme == "light" else "#f0f0f0"
            self.input_field = "#f0f0f0" if theme == "light" else "#1e1e1e"
            self.cutoff = cutoff
            self.kwargs = kwargs

        except Exception as e:
            print(f"Error initializing ChatApp: {e}")

    def compose(self) -> ComposeResult:
        try:
            self.CSS = f"""
            ChatApp {{
                background: {self.background};
            }}
            
            RichLog#chat_display {{
                border: round $primary;
                background: {self.background};
                color: {self.text};
                padding: 1 2;
            }}
        
            Input#input_field {{
                border: round $primary;
                background: {self.input_field};
                color: $text;
                padding: 1 2;
            }}
            """

            with VerticalScroll():
                yield RichLog(id="chat_display")
                yield Input(placeholder="Type your message...", id="input_field")
        except Exception as e:
            print(f"Error composing ChatApp: {e}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            user_message = event.value.strip()
            if user_message:
                self.chat_history.append({"role": "user", "content": user_message})

                user_text = Text()
                user_text.append("User: ", style="bold blue")
                user_text.append(f"{user_message}\n")

                self.query_one("#chat_display", RichLog).write(user_text)

                if self.provider:
                    response = self.provider.run(user_message, **self.kwargs)
                else:
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
                        **self.kwargs,
                    )
                    assistant_reply = response.choices[0].message["content"]
                    self.chat_history.append(
                        {"role": "assistant", "content": assistant_reply}
                    )
                    response = assistant_reply

                assistant_text = Text()
                assistant_text.append("Assistant: ", style="bold green")

                if "\n" in response:
                    for section in response.split("\n"):
                        assistant_text.append(f"{section}\n")
                else:
                    for i in range(0, len(response), self.cutoff):
                        assistant_text.append(f"{response[i:i+self.cutoff]}\n")

                self.query_one("#chat_display", RichLog).write(assistant_text)

                self.query_one("#input_field", Input).value = ""
        except Exception as e:
            print(f"Error processing input: {e}")


def chat(
    provider: Optional[Any] = None,
    messages: Union[str, list[dict]] = None,
    model: Optional[str] = "gpt-4o-mini",
    theme: Optional[Literal["light", "dark"]] = "dark",
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
    **kwargs,
):
    """Runs a Textual UI chat interface for a Chatbot, using either an Agents instance or the .completion() function.

    Parameters:
        provider: Optional[Any] = None,
        messages: Union[str, list[dict]] = None,
        model: Optional[str] = "gpt-4o-mini",
        theme: Optional[Literal["light", "dark"]] = "dark",
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
        **kwargs,
    """
    try:
        ChatApp(
            provider=provider,
            messages=messages,
            model=model,
            theme=theme,
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
            **kwargs,
        ).run()
    except Exception as e:
        print(f"Error running ChatApp: {e}")
