__all__ = ["app_cli"]


def app_cli():
    """
    CLI for the Zyx Model Application
    """

    import argparse
    from .main import app as chat_cli
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    parser = argparse.ArgumentParser(description="ZYX Application CLI")

    parser.add_argument(
        "--provider", type=str, help="The provider instance to handle the chat logic"
    )
    parser.add_argument(
        "--messages", type=str, help="The messages to send to the model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="The model to use for completions",
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default="dark",
        help="The theme to use for the chat interface",
    )
    parser.add_argument("--tools", nargs="+", help="The tools to use for completions")
    parser.add_argument(
        "--run-tools", type=bool, default=True, help="Whether to run the tools"
    )
    parser.add_argument(
        "--response-model",
        type=str,
        help="The Pydantic response model to use for completions",
    )
    parser.add_argument(
        "--mode", type=str, default="tools", help="The mode to use for completions"
    )
    parser.add_argument("--base-url", type=str, help="The base URL for the API")
    parser.add_argument("--api-key", type=str, help="The API key to use for the API")
    parser.add_argument(
        "--organization", type=str, help="The organization to use for the API"
    )
    parser.add_argument("--top-p", type=float, help="The top-p value for completions")
    parser.add_argument(
        "--temperature", type=float, help="The temperature value for completions"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="The maximum number of tokens for completions"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="The maximum number of retries for completions",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Whether to print verbose output"
    )
    parser.add_argument(
        "--background",
        type=str,
        default="midnight_black",
        help="Background color of the app",
    )
    parser.add_argument(
        "--text", type=str, default="steel_grey", help="Text color in the chat display"
    )
    parser.add_argument(
        "--input-field", type=str, default="ocean_blue", help="Color of the input field"
    )
    parser.add_argument(
        "--cutoff", type=int, default=95, help="The cutoff length for response display"
    )
    parser.add_argument("--help-menu", action="store_true", help="Show the help menu")

    args = parser.parse_args()

    if args.help_menu:

        class HelpMenu(App):
            def compose(self) -> ComposeResult:
                yield Static("ZYX Chat CLI Help Menu\n\n" + parser.format_help())

        HelpMenu().run()
    else:
        # Convert args to a dictionary and remove None values
        chat_args = {
            k: v for k, v in vars(args).items() if v is not None and k != "help_menu"
        }
        chat_cli(**chat_args)


def chat_cli():
    """
    CLI for the Chat Application
    """

    import argparse
    from .chat import chat as chat_cli
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    parser = argparse.ArgumentParser(description="ZYX Application CLI")

    parser.add_argument(
        "--provider", type=str, help="The provider instance to handle the chat logic"
    )
    parser.add_argument(
        "--messages", type=str, help="The messages to send to the model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="The model to use for completions",
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default="dark",
        help="The theme to use for the chat interface",
    )
    parser.add_argument("--tools", nargs="+", help="The tools to use for completions")
    parser.add_argument(
        "--run-tools", type=bool, default=True, help="Whether to run the tools"
    )
    parser.add_argument(
        "--response-model",
        type=str,
        help="The Pydantic response model to use for completions",
    )
    parser.add_argument(
        "--mode", type=str, default="tools", help="The mode to use for completions"
    )
    parser.add_argument("--base-url", type=str, help="The base URL for the API")
    parser.add_argument("--api-key", type=str, help="The API key to use for the API")
    parser.add_argument(
        "--organization", type=str, help="The organization to use for the API"
    )
    parser.add_argument("--top-p", type=float, help="The top-p value for completions")
    parser.add_argument(
        "--temperature", type=float, help="The temperature value for completions"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="The maximum number of tokens for completions"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="The maximum number of retries for completions",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Whether to print verbose output"
    )
    parser.add_argument(
        "--background",
        type=str,
        default="midnight_black",
        help="Background color of the app",
    )
    parser.add_argument(
        "--text", type=str, default="steel_grey", help="Text color in the chat display"
    )
    parser.add_argument(
        "--input-field", type=str, default="ocean_blue", help="Color of the input field"
    )
    parser.add_argument(
        "--cutoff", type=int, default=95, help="The cutoff length for response display"
    )
    parser.add_argument("--help-menu", action="store_true", help="Show the help menu")

    args = parser.parse_args()

    if args.help_menu:

        class HelpMenu(App):
            def compose(self) -> ComposeResult:
                yield Static("ZYX Chat CLI Help Menu\n\n" + parser.format_help())

        HelpMenu().run()
    else:
        # Convert args to a dictionary and remove None values
        chat_args = {
            k: v for k, v in vars(args).items() if v is not None and k != "help_menu"
        }
        chat_cli(**chat_args)
