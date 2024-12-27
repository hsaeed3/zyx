from __future__ import annotations

"""
zyx._utils

This module contains flags, logging, config & caching utilities for zyx.
All resources in this module are only meant to be used by the library
internally.
"""

# [Imports]
from typing import List, Dict, Any, Union
import builtins
import logging
import json
import os
from pathlib import Path
from pydantic import BaseModel
from functools import wraps

from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.traceback import install


# ===================================================================
# [Singletons]
# ===================================================================

# [Rich Console]
zyx_console : Console = Console()
# why not just install here
install(console = zyx_console)

# NOTE:
# zyx uses rich.print() at builtins level... 
# ill remove this probably, it just makes it prettier
builtins.print = rprint

# [Logger]
zyx_logger : logging.Logger = None


# ===================================================================
# [Constants]
# ===================================================================

# [Cache Directory]
# zyx uses `~/.cache/zyx` as its config & cache directory
ZYX_CACHE_DIR : Path = Path.home() / ".cache" / "zyx"

# [Config File]
# zyx uses `~/.config/zyx/config.json` as its config file
ZYX_CONFIG_FILE : Path = Path.home() / ".config" / "zyx" / "config.json"


# ===================================================================
# [Library Level Flags]
# ===================================================================

# [Verbosity]
zyx_verbose : bool = False
"""Modules will provide printed console outputs & simple information."""

# [Debug]
zyx_debug : bool = False
"""Modules will provide detailed debug information."""


# [Helpers]
def set_zyx_verbose(verbose : bool) -> None:
    """Sets the verbose flag."""
    global zyx_verbose
    zyx_verbose = verbose
    
def set_zyx_debug(debug : bool) -> None:
    """Sets the debug flag."""
    global zyx_debug
    zyx_debug = debug


# ===================================================================
# [Script Specific Flags]
# ===================================================================

# [Library Initialization Check]
zyx_initialized : bool = False
"""Internal flag that validates if the library has been initialized.
Used to prevent multiple initializations."""


# ===================================================================
# [Logging]
# ===================================================================

def get_logger() -> logging.Logger:
    """
    Retrieves the zyx logger instance & initializes it if it doesn't exist.
    """
    
    # Singleton Check
    global zyx_logger
    if zyx_logger is not None:
        return zyx_logger
    
    else:
        try:
            # Create a new logger instance
            zyx_logger = logging.getLogger("zyx")
            # Init at Quiet Level
            zyx_logger.setLevel(logging.WARNING)
            # Clear existing handlers
            if zyx_logger.hasHandlers():
                zyx_logger.handlers.clear()
                
            # Apply RichHandler
            handler = RichHandler(
                console = zyx_console,
                show_time = True,
                show_level = True,
                markup = True
            )
            handler.setLevel(logging.WARNING)
            zyx_logger.addHandler(handler)
        
        except Exception as e:
            raise RuntimeError(f"Failed to initialize zyx logger: {e}")
        # Return
        return zyx_logger


# [Logging Style Helpers]
class Styles:
    """
    Styles for zyx logging.
    
    These are helper functions that just make it easier to add styles
    to specific statements or phrases when creating log statements.

    In other words it looks pretty.
    """
    
    @staticmethod
    def zyx() -> str:
        """Library Name"""
        return "[bold italic light_sky_blue_3]zyx[/bold italic light_sky_blue_3]"
    
    @staticmethod
    def module(name : str) -> str:
        """Module and/or 'main' name"""
        return f"[bold light_coral]{name}[/bold light_coral]"
    
    @staticmethod
    def debug(message : str) -> str:
        return f"[dim]{message}[/dim]"
    
    
    # [Console Display Helpers] ====================================
    
    @staticmethod
    def typed_item(item : Union[
        List[Any], Dict[str, Any], BaseModel
    ]) -> Panel:
        """
        Displays specific types neatly in a panel.
        """
        if isinstance(item, list):
            # Create Table
            table = Table(show_header = True, header_style = "bold")
            # Add Items
            for i in item:
                table.add_row(i)
            # Return
            return Panel(table, title = "List", border_style = "dim")
        elif isinstance(item, dict):
            # Create Table
            table = Table(show_header=True, header_style="bold")
            table.add_column("Key", style="bold")
            table.add_column("Value")
            
            # Add Items
            for key, value in item.items():
                table.add_row(str(key), str(value))
            
            # Return
            return Panel(table, title="Dictionary", border_style="dim")
            
        elif isinstance(item, BaseModel):
            # Convert model to dict and display
            model_dict = item.model_dump()
            return Styles.display_item(model_dict)
        else:
            raise TypeError(f"Cannot display item of type {type(item)}")


# [Logger]
zyx_logger = get_logger()


# ===================================================================
# [Base Exception]
# ===================================================================

# Handled by rich.traceback automatically, just needs to be called
class ZyxException(Exception):
    """Base exception class used in zyx."""
    
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        # Log
        zyx_logger.error(message)
    
    
# ===================================================================
# [Config Utils]
# ===================================================================

# [Config File Validator]
def validate_zyx_config_file() -> Path:
    """
    Validates and/or initializes the zyx config file.
    """
    
    # Create config directory if it doesn't exist
    config_dir = ZYX_CONFIG_FILE.parent
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate if config file exists
    if not ZYX_CONFIG_FILE.exists():
        # Create config file
        ZYX_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Initialize with empty JSON object
        with open(ZYX_CONFIG_FILE, "w") as f:
            json.dump({}, f)
        
        # Validate if config file is writable
        if not os.access(ZYX_CONFIG_FILE, os.W_OK):
            from warnings import warn
            warn(f"Config file {ZYX_CONFIG_FILE} is not writable. No generations will be automatically saved.")
    # Return
    return ZYX_CONFIG_FILE


# [Get Config]
def get_zyx_config() -> dict:
    """
    Retrieves the zyx config file as a dictionary.
    """
    # Validate config file
    validate_zyx_config_file()
    # Load config file
    with open(ZYX_CONFIG_FILE, "r") as file:
        config = json.load(file)
    # Return
    return config


# [Set Config]
def set_zyx_config(key: str, value: any) -> None:
    """
    Sets a value in the zyx config file.
    If the key doesn't exist, it will be created.
    If the key exists, its value will be updated.

    Args:
        key: The config key to set
        value: The value to set for the key
    """
    # Validate config file
    validate_zyx_config_file()
    
    # Load existing config
    config = get_zyx_config()
    
    # Update value
    config[key] = value
    
    # Write updated config
    with open(ZYX_CONFIG_FILE, "w") as file:
        json.dump(config, file)


# ===================================================================
# [Core Util Functions]
# ===================================================================

# [Cache Validator]
def validate_zyx_cache_dir() -> Path:
    """
    Validates and/or initializes the zyx cache directory.
    """
    
    # Validate if cache directory exists
    if not ZYX_CACHE_DIR.exists():
        # Create cache directory
        ZYX_CACHE_DIR.mkdir(parents = True, exist_ok = True)
        # Validate if cache directory is writable
        if not ZYX_CACHE_DIR.is_writable():
            from warnings import warn
            warn(f"Cache directory {ZYX_CACHE_DIR} is not writable. No generations will be automatically saved.")
    # Return
    return ZYX_CACHE_DIR
    

# [Initializer]
def initialize_zyx():
    """
    Initializes the zyx library by ensuring resources & cache
    directory.
    """ 
    # Initialization Check
    global zyx_initialized
    if zyx_initialized:
        return
    
    # Get Base Logger
    global zyx_logger
    if zyx_logger is None:
        zyx_logger = get_logger()
    
    # Retrieve Global Flags
    global zyx_verbose
    global zyx_debug
    
    # Validate Cache Directory & Config File
    try:
        validate_zyx_cache_dir()
        validate_zyx_config_file()
    except Exception as e:
        raise ZyxException(f"Failed to validate cache directory @ [bold]{ZYX_CACHE_DIR}[/bold]: {e}")
    
    # Set Initialized Flag
    zyx_initialized = True
    
    