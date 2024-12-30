from __future__ import annotations

"""
zyx.core.litellm_resource

This module provides a singleton resource for the litellm library &
applies a few of the libraries flags.
"""

# [Imports]
from importlib import import_module

from zyx.core import utils


# ===================================================================
# [Singleton]
# ===================================================================

# [Resource]
litellm_resource : LiteLLMResource = None


# ===================================================================
# [LiteLLM Resource]
# ===================================================================

class LiteLLMResource:
    
    """
    A dynamic library wide singleton resource for the litellm library.
    
    LiteLLM initializes most its primary modules & functions at the library
    initialization level, so this resource helps both in lazy loading & only loading in one
    instance of the library.
    """
    
    litellm : any = None
    
    # [Initializer]
    def __new__(cls) -> LiteLLMResource:
        """Initializes the singleton instance of the LiteLLMResource."""
        
        if not hasattr(cls, '_instance'):
            try:
                # Import LiteLLM lazily
                if cls.litellm is None:
                    cls.litellm = import_module("litellm")
                    
                    # =======================================================
                    # LITELLM CONFIG FLAGS
                    # =======================================================
                    
                    # Configure LiteLLM flags for optimization
                    cls.litellm.drop_params = True  # Drop unused params
                    cls.litellm.modify_params = True  # Allow param modification
                    
                    if utils.zyx_debug:
                        utils.logger.debug(
                            f"Created {utils.Styles.module('LiteLLM')} resource singleton successfully."
                        )
                
                # Create singleton instance
                cls._instance = super().__new__(cls)
                
                # Set global reference
                global litellm_resource
                litellm_resource = cls._instance
                
            except Exception as e:
                utils.ZyxException(f"Failed to initialize LiteLLM resource: {e}")
        else:
            if utils.zyx_debug:
                utils.logger.debug(
                    f"Loaded prexisting {utils.Styles.module('LiteLLM')} resource successfully."
                )
                
        return cls._instance
            

# ===================================================================
# [Helper Function]
# ===================================================================

def get_litellm_resource() -> LiteLLMResource:
    """Returns the singleton instance of the LiteLLMResource."""
    
    global litellm_resource
    
    if litellm_resource is None:
        litellm_resource = LiteLLMResource()
    else:
        if utils.zyx_debug:
            utils.logger.debug(
                f"Loaded prexisting {utils.Styles.module('LiteLLM')} resource successfully."
            )
        
    return litellm_resource