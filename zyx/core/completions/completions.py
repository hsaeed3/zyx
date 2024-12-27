from __future__ import annotations

# TODO:
# finish docstring

"""
zyx.core.completions.completions

zyx completions Base Resource
This module provides the pipeline for the .completions() method.

zyx completions are an extension over litellm.completion(), by providing
the following additional functionality:
- Use any model supported by litellm
- [Tool Usage]
    - Use any python function, pydantic model or dictionary as a tool
    - Automatically execute tool calls with `run_tools`
"""

# [Imports]
from typing import Any, Callable, Dict, List, Optional, Union, Literal, overload