from rich.console import Console
from zyx import Completions
from zyx._environ import Environment
from zyx.resources.types.completion_create_params import ZYX_DEFAULT_MODEL
from ._rich import console
from typing import Optional
import typer
from typer import Typer
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

app = Typer(add_completion=False)

@app.command(name = "model", help = "Change the default model (library wide)")
def set_default_model(model : str) -> None:
    Environment.change_default_model(model)
    console.print(f"⚛️ [bold cadet_blue] ZYX[/bold cadet_blue] Default model changed to: [bold green]{model}[/bold green]")

@app.command(name = "reset", help = "Reset zyx environment configuration")
def reset_model() -> None:
    if Environment.reset_config(verbose = True):
        console.print("⚛️ [bold cadet_blue] ZYX[/bold cadet_blue] Environment configuration [bold green]reset[/bold green].")
    else:
        console.print("⚛️ [bold cadet_blue] ZYX[/bold cadet_blue] Environment configuration [bold red]not reset[/bold red].")

@app.command(
    name="chat",
    help="Chat with a model",
)
def chat_app(
        message: Optional[str] = typer.Argument(None, help="Initial message to start the chat"),
        model: Optional[str] = typer.Option(None, help="Model to use for chat"),
        api_key: Optional[str] = typer.Option(None, help="API key for authentication"),
        base_url: Optional[str] = typer.Option(None, help="Base URL for API requests"),
        organization: Optional[str] = typer.Option(None, help="Organization for API requests"),
        temperature: Optional[float] = typer.Option(None, help="Temperature for response generation"),
        system_prompt: Optional[str] = typer.Option(None, help="System prompt to set context"),
        max_tokens: Optional[int] = typer.Option(None, help="Maximum number of tokens in the response"),
) -> None:
    console = Console()

    if model is None:
        model = ZYX_DEFAULT_MODEL

    try:
        client = Completions(
            api_key=api_key or None,
            base_url=base_url or None,
            organization=organization or None,
        )
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Temperature: {temperature}[/dim]")
    console.print(f"[dim]Max tokens: {max_tokens}[/dim]\n")
    console.print(f"[italic]Run 'zyx chat --help' for more information.[/italic]")
    console.print(f"[italic]Type 'exit', 'quit', or 'q' to quit.[/italic]\n")

    if message:
        messages.append({"role": "user", "content": message})
        process_message(client, console, messages, model, temperature, max_tokens)

    while True:
        print(messages)
        user_input = console.input("[bold green]> [/bold green]")

        if user_input in ["exit", "quit", "q"]:
            break

        messages.append({"role": "user", "content": user_input})
        process_message(client, console, messages, model, temperature, max_tokens)

def process_message(client, console, messages, model, temperature, max_tokens):
    response = client.completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        progress_bar=False
    )

    assistant_response = ""

    for chunk in response:
        assistant_response += chunk.choices[0].delta.content or ""
        console.print(chunk.choices[0].delta.content or "", end="", style="bold green")

    console.print()

    messages.append({"role": "assistant", "content": assistant_response})


@app.command(
    name="image",
    help="Generate an image based on a prompt",
)
def image_app(
    prompt: str = typer.Argument(..., help="Prompt for image generation"),
    model: str = typer.Option("dall-e-3", help="Model to use for image generation"),
    api_key: Optional[str] = typer.Option(None, help="API key for authentication"),
    image_size: str = typer.Option("landscape_4_3", help="Size of the generated image"),
    num_inference_steps: int = typer.Option(26, help="Number of inference steps"),
    guidance_scale: float = typer.Option(3.5, help="Guidance scale for image generation"),
    enable_safety_checker: bool = typer.Option(False, help="Enable safety checker"),
    size: str = typer.Option("1024x1024", help="Size of the generated image (for DALL-E models)"),
    quality: str = typer.Option("standard", help="Quality of the generated image (for DALL-E models)"),
    n: int = typer.Option(1, help="Number of images to generate"),
    display: bool = typer.Option(False, help="Display the generated image"),
    optimize_prompt: bool = typer.Option(False, help="Optimize the prompt before generation"),
    optimize_prompt_model: str = typer.Option("openai/gpt-4o-mini", help="Model to use for prompt optimization"),
):
    from .multimodal import image

    console = Console()
    try:
        result = image(
            prompt=prompt,
            model=model,
            api_key=api_key,
            image_size=image_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            enable_safety_checker=enable_safety_checker,
            size=size,
            quality=quality,
            n=n,
            display=display,
            optimize_prompt=optimize_prompt,
            optimize_prompt_model=optimize_prompt_model,
        )
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command(
    name="audio",
    help="Generate audio based on a prompt",
)
def audio_app(
    prompt: str = typer.Argument(..., help="Prompt for audio generation"),
    model: str = typer.Option("tts-1", help="Model to use for audio generation"),
    voice: str = typer.Option("alloy", help="Voice to use for audio generation"),
    api_key: Optional[str] = typer.Option(None, help="API key for authentication"),
    base_url: Optional[str] = typer.Option(None, help="Base URL for API requests"),
    filename: Optional[str] = typer.Option(None, help="Filename to save the generated audio"),
    play: bool = typer.Option(False, help="Play the generated audio"),
):
    from .multimodal import audio

    console = Console()
    try:
        result = audio(
            prompt=prompt,
            model=model,
            voice=voice,
            api_key=api_key,
            base_url=base_url,
            filename=filename,
            play=play,
        )
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console = Console()

        zyx_art = r"""
 ______     __  __     __  __
/\___  \   /\ \_\ \   /\_\_\_\
\/_/  /__  \ \____ \  \/_/\_\/_
  /\_____\  \/\_____\   /\_\/\_\
  \/_____/   \/_____/   \/_/\/_/
        """

        info_text = """
Available commands:
"""

        commands = [
            ("zyx chat", "Start an interactive chat session"),
            ("zyx image", "Generate images from text prompts"),
            ("zyx audio", "Generate audio from text prompts"),
        ]

        for command, description in commands:
            info_text += f"• {command}: {description}\n"

        info_text += """
Run `zyx <command> --help`
for more information on each command.
        """

        panel = Panel(
            Text(zyx_art + info_text, justify="center"),
            title="zyx CLI",
            expand=True,
            border_style="cyan",
            highlight=True
        )

        console.print(panel, width=console.width)

if __name__ == "__main__":
    app()
