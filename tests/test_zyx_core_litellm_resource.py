"""
LiteLLM Resource Tests
"""

import pytest
from zyx.core._utils import set_zyx_debug
from zyx.core.litellm_resource import get_litellm_resource

set_zyx_debug(True)


# [Singleton Test]
def test_zyx_core_litellm_resource_singleton() -> None:
    """Tests the singleton instance of the LiteLLMResource."""
    
    # Get Singletons
    litellm_resource_1 = get_litellm_resource()
    litellm_resource_2 = get_litellm_resource()
    
    # Check
    assert litellm_resource_1 is litellm_resource_2
    assert litellm_resource_1 is not None
    assert litellm_resource_2 is not None
    
    
# [Completion Test]
def test_zyx_core_litellm_resource_completion() -> None:
    """Runs a litellm.completion() using the LiteLLMResource singleton."""
    
    # Get Resource
    litellm_resource = get_litellm_resource()
    
    # Run Completion
    completion = litellm_resource.litellm.completion(
        model = "gpt-4o",
        messages = [{"role": "user", "content": "Hello, how are you?"}],
        mock_response = "Hello, I'm fine, thank you!"
    )
    
    # Display Completion
    print(completion)
    
    assert completion.choices[0].message.content == "Hello, I'm fine, thank you!"


# ===================================================================
# [Run]
# ===================================================================

if __name__ == "__main__":
    test_zyx_core_litellm_resource_singleton()
    test_zyx_core_litellm_resource_completion()