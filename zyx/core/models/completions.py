from __future__ import annotations

"""
Completions Config & Parameter Models
"""

from pydantic import BaseModel, ConfigDict


# ===================================================================
# [Completion Pipeline Config]
# ===================================================================

class CompletionConfig(BaseModel):
    
    """
    Completion Pipeline Configuration Model
    """
    
    