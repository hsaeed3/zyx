from __future__ import annotations

__all__ = [
    "Environment",

    # pre-init exports
    "ZYX_DEFAULT_MODEL"
]

# hammad saeed - 2024
# zyx._Environment
# internal library helpers
# Environment handling

from ..resources.types import completion_create_params as params
from .utils import console
from enum import Enum
from pathlib import Path
import os
from typing import Literal, Union
from rich.prompt import Prompt


# Environment helpers
# Used for controlling new default service params

# defaults enum model
class ZyxDefaults(Enum):
    """Default environment values."""

    # Library Default LLM
    ZYX_DEFAULT_MODEL = "gpt-4o-mini"


ZyxDefaultsKeys = Literal["ZYX_DEFAULT_MODEL"]


# Environment util
class Environment:
    """Environment utility class for managing Zyx configuration."""

    # Environments
    defaults = ZyxDefaults

    # ensure zyx config dir exists
    @staticmethod
    def ensure_zyx_config_dir(verbose : bool = False, clear : bool = False) -> None:
        """Ensure Zyx config directory exists and initialize with defaults if needed.

        Example:
            >>> Environment.ensure_zyx_config_dir(verbose=True)
            >>> Environment.ensure_zyx_config_dir(clear=True)

        Parameters:
            verbose: Whether to print status messages
            clear: Whether to clear existing config directory

        Returns:
            None
        """

        # get config dir from user's home directory
        config_dir = Path.home() / ".zyx"

        if clear:
            # ensure it exists
            config_dir.mkdir(parents=True, exist_ok=True)
            # ENSURE WITH USER (WE DONT JUST DELETE SHIT)
            user_response = Prompt().ask(f"⚠️ [bold]Are you sure you want to clear the [red]zyx[/red] config directory @ [dim]{config_dir}[/dim]?[/bold]", default=False)
            if user_response.lower() not in ['y', 'yes']:
                return False

            # clear it
            for file in config_dir.iterdir():
                file.unlink()

        # ensure it exists
        config_dir.mkdir(parents=True, exist_ok=True)

        # ensure default Environment file exists
        Environment_file = config_dir / ".Environment"

        # if not exist; write all values in defaults enum
        if not Environment_file.exists():

            if verbose:
                console.print(f"⚛️ [bold green]Creating zyx config directory @ [dim]{config_dir}[/dim]...[/bold green]")

            # create dir if not exists
            Environment_file.parent.mkdir(parents=True, exist_ok=True)

            # create Environment file if not exists
            Environment_file.touch()

            # write all defaults
            for pair in ZyxDefaults:
                Environment.set_Environment(pair.name, pair.value)

        else:
            if verbose:
                console.print(f"✅ [bold green]zyx config directory already exists @ [dim]{config_dir}[/dim][/bold green]")

        return True

    # get Environment value
    @staticmethod
    def get_Environment(key: ZyxDefaultsKeys) -> str:
        """Get environment value for a given key.

        Example:
            >>> model = Environment.get_Environment("ZYX_DEFAULT_MODEL")

        Parameters:
            key: Environment variable key to retrieve

        Returns:
            str: Value of the environment variable
        """

        # ensure Environment file exists
        Environment.ensure_zyx_config_dir()

        # get Environment file
        Environment_file = Path.home() / ".zyx" / ".Environment"

        # read Environment file
        with Environment_file.open("r") as f:
            for line in f:
                if line.startswith(key):
                    return line.split("=")[1].strip()

        # get from os
        return os.getEnvironment(key)


    # set / update Environment value
    @staticmethod
    def set_Environment(key: ZyxDefaultsKeys, value: str) -> str:
        """Set or update environment value for a given key.

        Example:
            >>> Environment.set_Environment("ZYX_DEFAULT_MODEL", "gpt-4")

        Parameters:
            key: Environment variable key to set
            value: Value to set for the key

        Returns:
            str: The value that was set
        """

        # ensure Environment file exists
        Environment.ensure_zyx_config_dir()

        # get Environment file
        Environment_file = Path.home() / ".zyx" / ".Environment"

        # read current contents
        lines = []
        if Environment_file.exists():
            with Environment_file.open("r") as f:
                lines = f.readlines()

        # update or add the key-value pair
        updated = False
        with Environment_file.open("w", newline='') as f:
            for line in lines:
                if line.startswith(key):
                    f.write(f"{key}={value}\n")
                    updated = True
                else:
                    f.write(line)
            if not updated:
                f.write(f"{key}={value}\n")

        # return value
        return value
    

    @staticmethod
    def change_default_model(model : Union[str, params.ChatModel]) -> None:
        """Change the default model used by Zyx.

        Example:
            >>> Environment.change_default_model("gpt-4")

        Parameters:
            model: New default model name or ChatModel instance

        Returns:
            None
        """

        Environment.set_Environment("ZYX_DEFAULT_MODEL", model)


    @staticmethod
    def reset_config(verbose: bool = False) -> None:
        """Reset all environment variables to defaults.

        Example:
            >>> Environment.reset_config(verbose=True)

        Parameters:
            verbose: Whether to print status messages

        Returns:
            None
        """

        # Ensure the Environment file is updated with all defaults
        Environment.ensure_zyx_config_dir(clear=True, verbose=verbose)


# PRE-INIT Environment FOR EXPORTS

# default llm (model)
ZYX_DEFAULT_MODEL = Environment.get_Environment("ZYX_DEFAULT_MODEL")


# test
if __name__ == "__main__":

    print(ZYX_DEFAULT_MODEL)

    Environment.ensure_zyx_config_dir()

    Environment.set_Environment("ZYX_DEFAULT_MODEL", "gpt-4o-mini")

    print(Environment.get_Environment("ZYX_DEFAULT_MODEL"))

    Environment.set_Environment("ZYX_DEFAULT_MODEL", "gpt-4o")

    print(Environment.get_Environment("ZYX_DEFAULT_MODEL"))

    Environment.reset(verbose=True)

    print(Environment.get_Environment("ZYX_DEFAULT_MODEL"))

    # Re-fetch the default model after reset
    ZYX_DEFAULT_MODEL = Environment.get_Environment("ZYX_DEFAULT_MODEL")

    print(ZYX_DEFAULT_MODEL)

    Environment.set_Environment("ZYX_DEFAULT_MODEL", "gpt-4o-mini")

    print("[bold red] Hello World! [/bold red]")
