from textual.widgets import Input, RichLog, Button, Tabs, Tab, Static
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from typing import Union, Optional, List, Callable, Literal
from pydantic import BaseModel


from ...lib.utils.logger import get_logger
from ...client import Client, completion, InstructorMode


logger = get_logger("app")


COLOR_MAP = {
    "black": "#000000",
    "white": "#FFFFFF",
    "red": "#FF0000",
    "green": "#008000",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "silver": "#C0C0C0",
    "deep_blue": "#001f3f",
    "ocean_blue": "#0074D9",
    "sunset_orange": "#FF851B",
    "twilight_purple": "#6F42C1",
    "forest_green": "#2ECC40",
    "midnight_black": "#111111",
    "crimson_red": "#DC143C",
    "royal_gold": "#FFD700",
    "peach": "#FFDAB9",
    "lavender": "#E6E6FA",
    "teal": "#008080",
    "coral": "#FF7F50",
    "mustard_yellow": "#FFDB58",
    "powder_blue": "#B0E0E6",
    "sage_green": "#B2AC88",
    "blush": "#FF6F61",
    "steel_grey": "#7A8B8B",
    "ice_blue": "#AFEEEE",
    "burnt_sienna": "#E97451",
    "plum": "#DDA0DD",
    "emerald_green": "#50C878",
    "ruby_red": "#E0115F",
    "sapphire_blue": "#0F52BA",
    "amethyst_purple": "#9966CC",
    "topaz_yellow": "#FFC87C",
    "turquoise": "#40E0D0",
    "rose_gold": "#B76E79",
    "olive_green": "#808000",
    "burgundy": "#800020",
    "navy_blue": "#000080",
    "mauve": "#E0B0FF",
    "chartreuse": "#7FFF00",
    "terracotta": "#E2725B",
    "indigo": "#4B0082",
    "periwinkle": "#CCCCFF",
    "maroon": "#800000",
    "cerulean": "#007BA7",
    "ochre": "#CC7722",
    "slate_gray": "#708090",
    "mint_green": "#98FF98",
    "salmon": "#FA8072",
    "tangerine": "#F28500",
    "taupe": "#483C32",
    "aquamarine": "#7FFFD4",
    "mahogany": "#C04000",
    "fuchsia": "#FF00FF",
    "azure": "#007FFF",
    "lilac": "#C8A2C8",
    "vermilion": "#E34234",
    "ivory": "#FFFFF0",
}

ColorName = Literal[
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "silver",
    "deep_blue",
    "ocean_blue",
    "sunset_orange",
    "twilight_purple",
    "forest_green",
    "midnight_black",
    "crimson_red",
    "royal_gold",
    "peach",
    "lavender",
    "teal",
    "coral",
    "mustard_yellow",
    "powder_blue",
    "sage_green",
    "blush",
    "steel_grey",
    "ice_blue",
    "burnt_sienna",
    "plum",
    "emerald_green",
    "ruby_red",
    "sapphire_blue",
    "amethyst_purple",
    "topaz_yellow",
    "turquoise",
    "rose_gold",
    "olive_green",
    "burgundy",
    "navy_blue",
    "mauve",
    "chartreuse",
    "terracotta",
    "indigo",
    "periwinkle",
    "maroon",
    "cerulean",
    "ochre",
    "slate_gray",
    "mint_green",
    "salmon",
    "tangerine",
    "taupe",
    "aquamarine",
    "mahogany",
    "fuchsia",
    "azure",
    "lilac",
    "vermilion",
    "ivory",
]


class ZyxApp(App):
    CSS = """
    Screen {
        background: $background;
        height: 100vh;
    }

    #chat_display {
        border: heavy $primary;
        background: $surface;
        color: $text;
        padding: 1 2;
        height: 1fr;
    }

    #input_field {
        dock: bottom;
        width: 100%;
        margin: 1 0;
    }

    #button_row {
        dock: bottom;
        width: 100%;
        height: auto;
        margin-bottom: 5;
    }

    #send_button, #clear_button {
        width: 50%;
        margin: 0;
        height: auto;
        display: block;
    }

    #params_container {
        height: 100%;
        padding: 1;
    }

    .param_input {
        margin-bottom: 1;
    }

    #save_button {
        dock: bottom;
    }
    
    #chat_content, #params_content {
        height: 100%;
    }

    Tabs {
        dock: top;
    }

    .param_label {
        width: 30%;
        padding-right: 1;
        text-align: right;
        margin-bottom: 1;
    }

    .param_input {
        width: 70%;
        margin-bottom: 1;
    }

    #params_content Horizontal {
        height: auto;
        margin-bottom: 1;
    }

    #save_button {
        margin-top: 2;
    }
    """

    def __init__(
        self,
        messages: Union[str, list[dict]] = None,
        model: Optional[str] = "gpt-4o-mini",
        theme: Optional[Literal["light", "dark"]] = "dark",
        tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
        run_tools: Optional[bool] = True,
        response_model: Optional[BaseModel] = None,
        mode: Optional[InstructorMode] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = 3,
        verbose: Optional[bool] = False,
        background: Optional[Union[str, ColorName]] = None,
        text: Optional[Union[str, ColorName]] = None,
        input_field: Optional[Union[str, ColorName]] = None,
        cutoff: Optional[int] = 95,
        **kwargs,
    ):
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
        self.background = COLOR_MAP.get(background, background)
        self.text = COLOR_MAP.get(text, text)
        self.input_field = COLOR_MAP.get(input_field, input_field)
        self.cutoff = cutoff
        self.kwargs = kwargs
        self.theme = theme

        self.params = {
            "model": model or "openai/gpt-4o-mini",
            "max_tokens": max_tokens or 1000,
            "temperature": temperature or 0.7,
            "instruction": "You are a helpful assistant.",
        }

    def compose(self) -> ComposeResult:
        yield Tabs(
            Tab("Chat", id="chat_tab"),
            Tab("Parameters", id="params_tab"),
        )

        with Vertical(id="chat_content"):
            yield VerticalScroll(RichLog(id="chat_display"))
            yield Input(placeholder="Type your message...", id="input_field")
            with Horizontal(id="button_row"):
                yield Button(label="Send", id="send_button")
                yield Button(label="Clear", id="clear_button")

        with Vertical(id="params_content"):
            yield VerticalScroll(
                Horizontal(
                    Static("Model:", classes="param_label"),
                    Input(id="model_input", classes="param_input"),
                ),
                Horizontal(
                    Static("Max Tokens:", classes="param_label"),
                    Input(id="max_tokens_input", classes="param_input"),
                ),
                Horizontal(
                    Static("Temperature:", classes="param_label"),
                    Input(id="temperature_input", classes="param_input"),
                ),
                Horizontal(
                    Static("Instruction:", classes="param_label"),
                    Input(id="instruction_input", classes="param_input"),
                ),
                Static("", classes="spacer"),
                Button(label="Save", id="save_button", variant="primary"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_button":
            self.submit_message()
        elif event.button.id == "clear_button":
            self.clear_messages()
        elif event.button.id == "save_button":
            self.save_params()

    def on_mount(self) -> None:
        self.set_theme(self.background, self.text, self.input_field)
        self.query_one("#chat_content").display = True
        self.query_one("#params_content").display = False
        self.load_params()

    def set_theme(self, background, text, input_field):
        self.dark = self.theme != "light"
        self.styles.background = background or ("#1e1e1e" if self.dark else "#f0f0f0")
        self.styles.color = text or ("#ffffff" if self.dark else "#000000")
        self.query_one("#input_field").styles.background = input_field or (
            "#2e2e2e" if self.dark else "#ffffff"
        )

    def clear_messages(self):
        self.query_one("#chat_display", RichLog).clear()

    def save_params(self):
        self.params["model"] = self.query_one("#model_input").value or self.model
        self.params["max_tokens"] = (
            int(self.query_one("#max_tokens_input").value or 0) or None
        )
        self.params["temperature"] = (
            float(self.query_one("#temperature_input").value or 0) or None
        )
        self.params["instruction"] = self.query_one("#instruction_input").value

        # Update the class attributes
        self.model = self.params["model"]
        self.max_tokens = self.params["max_tokens"]
        self.temperature = self.params["temperature"]

        # Update the client with new parameters
        self.client.model = self.model
        self.client.max_tokens = self.max_tokens
        self.client.temperature = self.temperature

        logger.info(f"Saved parameters: {self.params}")
        self.notify("Parameters saved successfully!")

    def submit_message(self):
        from rich.text import Text

        user_message = self.query_one("#input_field", Input).value.strip()
        if user_message:
            self.chat_history.append({"role": "user", "content": user_message})

            user_text = Text()
            user_text.append("User: ", style="bold blue")
            user_text.append(f"{user_message}\n")

            self.query_one("#chat_display", RichLog).write(user_text)

            response = completion(
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
            assistant_reply = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": assistant_reply})

            assistant_text = Text()
            assistant_text.append("Assistant: ", style="bold green")

            if "\n" in assistant_reply:
                for section in assistant_reply.split("\n"):
                    assistant_text.append(f"{section}\n")
            else:
                for i in range(0, len(assistant_reply), self.cutoff):
                    assistant_text.append(f"{assistant_reply[i:i+self.cutoff]}\n")

            self.query_one("#chat_display", RichLog).write(assistant_text)
            self.query_one("#input_field", Input).value = ""

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.submit_message()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab.id == "chat_tab":
            self.query_one("#chat_content").display = True
            self.query_one("#params_content").display = False
        elif event.tab.id == "params_tab":
            self.query_one("#chat_content").display = False
            self.query_one("#params_content").display = True

    def load_params(self):
        self.query_one("#model_input").value = self.params["model"]
        self.query_one("#max_tokens_input").value = str(self.params["max_tokens"])
        self.query_one("#temperature_input").value = str(self.params["temperature"])
        self.query_one("#instruction_input").value = self.params["instruction"]


def terminal(
    messages: Union[str, list[dict]] = None,
    model: Optional[str] = "gpt-4o-mini",
    theme: Optional[Literal["light", "dark"]] = "dark",
    tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
    run_tools: Optional[bool] = True,
    response_model: Optional[BaseModel] = None,
    mode: Optional[InstructorMode] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    verbose: Optional[bool] = False,
    background: Optional[Union[str, ColorName]] = None,
    text: Optional[Union[str, ColorName]] = None,
    input_field: Optional[Union[str, ColorName]] = None,
    cutoff: Optional[int] = 95,
    **kwargs,
):
    """Runs an easy to use CLI interface for a Chatbot using the completion function."""

    try:
        ZyxApp(
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
            background=background,
            text=text,
            input_field=input_field,
            cutoff=cutoff,
            **kwargs,
        ).run()
    except Exception as e:
        print(f"Error running ZyxApp: {e}")


if __name__ == "__main__":
    terminal()
