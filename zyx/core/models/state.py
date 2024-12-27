from __future__ import annotations

"""
Message Thread State Models
"""

# [Imports]
from typing import Sequence, Dict, Any, Union, Optional
from pydantic import BaseModel, ConfigDict


# ===================================================================
# [Message Thread State]
# ===================================================================

class State(BaseModel):
    
    """
    Base State Model for `zyx` completions, agents and other
    LLM based modules.
    """