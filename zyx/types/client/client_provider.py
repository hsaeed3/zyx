"""
zyx.types.client.client_provider

Contains the client `provider` type, which is either one of:
- OpenAI
- LiteLLM
"""

from __future__ import annotations

# [Imports]
from typing import Literal


# ===================================================================
# [Client Provider Type]
# ===================================================================


ClientProvider = Literal["openai", "litellm"]
"""The type for the client provider."""
