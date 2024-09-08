from textual.widgets import Input, RichLog, Button, Tabs, Tab, RadioSet, RadioButton, Static
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from typing import Any, Union, Optional, List, Callable, Literal
from pydantic import BaseModel
from .completion import ClientModeParams
from loguru import logger
import os


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

    #chat_content, #params_content {
        display: none;
    }

    Tabs {
        dock: top;
    }

    #image_content, #speak_content, #transcribe_content {
        height: 100%;
        display: none;
    }

    .tab_content {
        padding: 1;
    }

    #image_display, #audio_display, #transcription_display {
        border: heavy $primary;
        background: $surface;
        color: $text;
        padding: 1 2;
        height: 1fr;
        margin-bottom: 1;
    }

    #image_model_select {
        margin-bottom: 1;
    }

    RadioSet {
        layout: horizontal;
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }

    RadioButton {
        margin-right: 1;
    }

    #params_content {
        padding: 1 2;
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
        background: Optional[Union[str, ColorName]] = None,
        text: Optional[Union[str, ColorName]] = None,
        input_field: Optional[Union[str, ColorName]] = None,
        cutoff: Optional[int] = 95,
        **kwargs,
    ):
        try:
            from .completion import CompletionClient

            super().__init__()
            self.client = CompletionClient()
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
            self.background = COLOR_MAP.get(
                kwargs.get("background", None), kwargs.get("background", None)
            )
            self.text = COLOR_MAP.get(
                kwargs.get("text", None), kwargs.get("text", None)
            )
            self.input_field = COLOR_MAP.get(
                kwargs.get("input_field", None), kwargs.get("input_field", None)
            )
            self.cutoff = cutoff
            self.kwargs = kwargs
            self.theme = theme
            self.audio_data = None
            self.transcription = None

            self.params = {
                "model": model or "openai/gpt-4o-mini",
                "max_tokens": max_tokens or 1000,  # Set your default max_tokens
                "temperature": temperature or 0.7,  # Set your default temperature
                "instruction": "You are a helpful assistant.",  # Set your default instruction if needed
            }

        except Exception as e:
            print(f"Error initializing ChatApp: {e}")

    def compose(self) -> ComposeResult:
        yield Tabs(
            Tab("Chat", id="chat_tab"),
            Tab("Parameters", id="params_tab"),
            Tab("Image", id="image_tab"),
            Tab("Speak", id="speak_tab"),
            Tab("Transcribe", id="transcribe_tab"),
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

        with Vertical(id="image_content", classes="tab_content"):
            yield VerticalScroll(RichLog(id="image_display"))
            yield RadioSet(
                RadioButton("DALL-E 2", id="dall-e-2"),
                RadioButton("DALL-E 3", id="dall-e-3", value=True),
                RadioButton("Flux Dev", id="flux-dev"),
                RadioButton("Flux Realism", id="flux-realism"),
                RadioButton("Flux Schnell", id="flux-schnell"),
                RadioButton("Flux Pro", id="flux-pro"),
                RadioButton("Flux LoRA", id="flux-lora"),
                RadioButton("Flux General", id="flux-general"),
                RadioButton("Aura", id="aura"),
                RadioButton("SD v3", id="sd-v3"),
                RadioButton("Fooocus", id="fooocus"),
                id="image_model_select",
            )
            yield Input(placeholder="Enter image prompt...", id="image_prompt")
            yield Button(label="Generate Image", id="generate_image_button")

        with Vertical(id="speak_content", classes="tab_content"):
            yield VerticalScroll(RichLog(id="audio_display"))
            yield Input(placeholder="Enter text to speak...", id="speak_text")
            yield Button(label="Generate Audio", id="generate_audio_button")
            yield Button(label="Play Audio", id="play_button")

        with Vertical(id="transcribe_content", classes="tab_content"):
            yield VerticalScroll(RichLog(id="transcription_display"))
            yield Button(label="Record Audio", id="record_button")
            yield Button(label="Transcribe", id="transcribe_button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_button":
            self.submit_message()
        elif event.button.id == "clear_button":
            self.clear_messages()
        elif event.button.id == "save_button":
            self.save_params()
        elif event.button.id == "generate_image_button":
            self.generate_image()
        elif event.button.id == "generate_audio_button":
            self.generate_audio()
        elif event.button.id == "play_button":
            self.play_audio()
        elif event.button.id == "record_button":
            self.record_audio()
        elif event.button.id == "transcribe_button":
            self.transcribe_audio()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "image_model_select":
            self.image_model = event.pressed.id

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
        if self.query_one("#model_input").value == "":
            self.params["model"] = self.model
        else:
            self.params["model"] = self.query_one("#model_input").value
        if self.query_one("#max_tokens_input").value == "":
            self.params["max_tokens"] = None
        else:
            self.params["max_tokens"] = int(self.query_one("#max_tokens_input").value)
        if self.query_one("#temperature_input").value == "":
            self.params["temperature"] = None
        else:
            self.params["temperature"] = float(
                self.query_one("#temperature_input").value
            )
        instruction = self.query_one("#instruction_input").value
        if instruction:
            self.swap_system_prompt(instruction)
        self.params["instruction"] = instruction

        # Update the class attributes
        self.model = self.params["model"]
        self.max_tokens = self.params["max_tokens"]
        self.temperature = self.params["temperature"]

        # Update the client with new parameters
        self.client.model = self.model
        self.client.max_tokens = self.max_tokens
        self.client.temperature = self.temperature

        logger.info(f"Saved parameters: {self.params}")

        # Notify the user
        self.notify("Parameters saved successfully!")

    def generate_response(self, user_message):
        try:
            from rich.text import Text

            if self.provider:
                response = self.provider.run(user_message, **self.kwargs)
            else:
                logger.info(
                    f"Sending message to {self.model} with {len(self.chat_history)} messages"
                )

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

        except Exception as e:
            print(f"Error generating response: {e}")

    def swap_system_prompt(self, instruction):
        from .utils.messages import swap_system_prompt

        instruction = {"role": "system", "content": instruction}
        self.chat_history = swap_system_prompt(
            messages=self.chat_history, system_prompt=instruction
        )

    def load_params(self):
        self.query_one("#model_input").value = self.params["model"]
        self.query_one("#max_tokens_input").value = str(self.params["max_tokens"])
        self.query_one("#temperature_input").value = str(self.params["temperature"])
        self.query_one("#instruction_input").value = self.params["instruction"]

    def submit_message(self):
        try:
            from rich.text import Text

            user_message = self.query_one("#input_field", Input).value.strip()
            if user_message:
                self.chat_history.append({"role": "user", "content": user_message})

                user_text = Text()
                user_text.append("User: ", style="bold blue")
                user_text.append(f"{user_message}\n")

                self.query_one("#chat_display", RichLog).write(user_text)

                if self.provider:
                    # Use the provider (Agents instance) to handle the message
                    response = self.provider.run(user_message, **self.kwargs)

                else:
                    logger.info(
                        f"Sending message to {self.model} with {len(self.chat_history)} messages"
                    )

                    # Use the completion client directly
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
                        {"role": "assistant",                         "content": assistant_reply}
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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            from rich.text import Text

            user_message = event.value.strip()
            if user_message:
                self.chat_history.append({"role": "user", "content": user_message})

                user_text = Text()
                user_text.append("User: ", style="bold blue")
                user_text.append(f"{user_message}\n")

                self.query_one("#chat_display", RichLog).write(user_text)

                if self.provider:
                    # Use the provider (Agents instance) to handle the message
                    response = self.provider.run(user_message, **self.kwargs)
                else:
                    # Use the completion client directly
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

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab.id == "chat_tab":
            self.query_one("#chat_content").display = True
            self.query_one("#params_content").display = False
        elif event.tab.id == "params_tab":
            self.query_one("#chat_content").display = False
            self.query_one("#params_content").display = True
        elif event.tab.id == "image_tab":
            self.query_one("#image_content").display = True
            self.query_one("#chat_content").display = False
            self.query_one("#params_content").display = False
            self.query_one("#speak_content").display = False
            self.query_one("#transcribe_content").display = False
        elif event.tab.id == "speak_tab":
            self.query_one("#speak_content").display = True
            self.query_one("#chat_content").display = False
            self.query_one("#params_content").display = False
            self.query_one("#image_content").display = False
            self.query_one("#transcribe_content").display = False
        elif event.tab.id == "transcribe_tab":
            self.query_one("#transcribe_content").display = True
            self.query_one("#chat_content").display = False
            self.query_one("#params_content").display = False
            self.query_one("#image_content").display = False
            self.query_one("#speak_content").display = False

    def generate_image(self):
        from .multimodal import image

        prompt = self.query_one("#image_prompt").value
        if prompt:
            try:
                result = image(
                    prompt=prompt, model=self.image_model, api_key=self.api_key
                )
                if isinstance(result, str):  # Error message
                    self.query_one("#image_display").write(f"Error: {result}")
                else:
                    if self.image_model in ["dall-e-2", "dall-e-3"]:
                        url = result.data[0].url
                    else:
                        url = result["images"][0]["url"]
                    self.query_one("#image_display").write(f"Image URL: {url}")
            except Exception as e:
                self.query_one("#image_display").write(
                    f"Error generating image: {str(e)}"
                )

    def generate_audio(self):
        from .multimodal import speak

        text = self.query_one("#speak_text").value
        if text:
            self.audio_data = speak(prompt=text, api_key=self.api_key, play=False)
            self.query_one("#audio_display").write(
                "Audio generated. Click 'Play Audio' to listen."
            )

    def play_audio(self):
        if self.audio_data:
            import sounddevice as sd

            audio_array, sample_rate = self.audio_data
            sd.play(audio_array, sample_rate)
            sd.wait()

    def record_audio(self):
        import sounddevice as sd
        import numpy as np

        duration = 5  # seconds
        sample_rate = 44100
        self.query_one("#transcription_display").write("Recording for 5 seconds...")
        recording = sd.rec(
            int(duration * sample_rate), samplerate=sample_rate, channels=1
        )
        sd.wait()
        self.audio_data = (recording, sample_rate)
        self.query_one("#transcription_display").write(
            "Recording finished. Click 'Transcribe' to process."
        )

    def transcribe_audio(self):
        from .multimodal import transcribe

        if self.audio_data:
            self.transcription = transcribe(
                api_key=self.api_key, record=False, file=self.audio_data
            )
            self.query_one("#transcription_display").write(
                f"Transcription: {self.transcription}"
            )
        else:
            self.query_one("#transcription_display").write(
                "No audio recorded. Please record audio first."
            )


def terminal(
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
    background: Optional[Union[str, ColorName]] = None,
    text: Optional[Union[str, ColorName]] = None,
    input_field: Optional[Union[str, ColorName]] = None,
    cutoff: Optional[int] = 95,
    **kwargs,
):
    """Runs an easy to use CLI interface for a Chatbot, using either an Agents instance or the .completion() function.

    Parameters:
        - provider (Optional[Any]): The provider instance (e.g., Agents) to handle the chat logic.
        - messages (Union[str, list[dict]]): The messages to send to the model.
        - model (Optional[str]): The model to use for completions.
        - tools (Optional[List[Union[Callable, dict, BaseModel]]]): The tools to use for completions.
        - run_tools (Optional[bool]): Whether to run the tools.
        - response_model (Optional[BaseModel]): The Pydantic response model to use for completions.
        - background (Optional[Union[str, ColorName]]): Background color of the app.
        - text (Optional[Union[str, ColorName]]): Text color in the chat display.
        - input_field (Optional[Union[str, ColorName]]): Color of the input field.
        - mode (Optional[str]): The mode to use for completions.
        - base_url (Optional[str]): The base URL for the API.
        - api_key (Optional[str]): The API key to use for the API.
        - organization (Optional[str]): The organization to use for the API.
        - top_p (Optional[float]): The top-p value for completions.
        - temperature (Optional[float]): The temperature value for completions.
        - max_tokens (Optional[int]): The maximum number of tokens for completions.
        - max_retries (Optional[int]): The maximum number of retries for completions.
        - verbose (Optional[bool]): Whether to print verbose output.
    """

    try:
        ZyxApp(
            provider=provider,
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
            background=background,
            text=text,
            input_field=input_field,
            cutoff=cutoff,
            **kwargs,
        ).run()
    except Exception as e:
        print(f"Error running ChatApp: {e}")
