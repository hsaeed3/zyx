# zyx._api_resource tests


import pytest
import logging
from zyx import _api_resource


logger = logging.getLogger("zyx")


def test_api_resource_litellm_resource():

    logger.setLevel(logging.DEBUG)

    # test litellm client init
    litellm = _api_resource.LiteLLMResource.get_litellm()
    assert litellm is not None
    logger.debug(f"litellm client: {litellm}")

    # test litellm client mock method
    assert litellm.completion(
        model = "gpt-4o",
        messages = [{"role": "user", "content": "Hello, world!"}],
        mock_response = "Hello, world!",
    ).choices[0].message.content == "Hello, world!"
    logger.debug(f"litellm client mock method: {litellm.completion(model='gpt-4o', messages=[{'role': 'user', 'content': 'Hello, world!'}], mock_response='Hello, world!')}")

    # test singleton
    client_1 = _api_resource.LiteLLMResource.get_litellm()
    client_2 = _api_resource.LiteLLMResource.get_litellm()
    assert client_1 is client_2
    logger.debug(f"litellm client singleton: {client_1} == {client_2}")


def test_api_resource_init():

    logger.setLevel(logging.DEBUG)

    # test api resource inits w/ none
    api_resource = _api_resource.APIResource()
    assert api_resource is not None
    assert api_resource.client is None
    logger.debug(f"api resource inits w/ none: {api_resource}")

    # -- LiteLLM provider tests --
    logger.debug("LiteLLM provider tests for zyx._api_resource.APIResource:")
    from litellm import LiteLLM

    # test api resource inits w/ litellm string provider
    api_resource = _api_resource.APIResource(provider="litellm")
    assert api_resource is not None
    assert isinstance(api_resource.client, LiteLLM)
    logger.debug(f"api resource inits w/ litellm string: {api_resource.client}")

    # test api resource inits w/ litellm instance
    api_resource = _api_resource.APIResource(client=LiteLLM())
    assert api_resource is not None
    assert isinstance(api_resource.client, LiteLLM)
    logger.debug(f"api resource inits w/ litellm instance: {api_resource}")

    # -- OpenAI provider tests --
    logger.debug("OpenAI provider tests for zyx._api_resource.APIResource:")
    from openai import OpenAI

    # test api resource inits w/ openai string
    api_resource = _api_resource.APIResource(provider="openai")
    assert api_resource is not None
    assert isinstance(api_resource.client, OpenAI)
    logger.debug(f"api resource inits w/ openai string: {api_resource}")

    # test api resource inits w/ openai instance
    api_resource = _api_resource.APIResource(client=OpenAI())
    assert api_resource is not None
    assert isinstance(api_resource.client, OpenAI)
    logger.debug(f"api resource inits w/ openai instance: {api_resource}")

    # test api resource inits w/ openai instance
    api_resource = _api_resource.APIResource(client=OpenAI())
    assert api_resource is not None
    assert isinstance(api_resource.client, OpenAI)
    logger.debug(f"api resource inits w/ openai instance: {api_resource}")


def test_api_resource_instructor_patch():

    logger.setLevel(logging.DEBUG)

    from openai import OpenAI

    # create client w/ openai
    client = _api_resource.APIResource(provider="openai")
    assert client is not None
    assert isinstance(client.client, OpenAI)
    logger.debug(f"api resource inits w/ openai instance: {client}")

    # patch client
    # use mode: "tool_call" (instructor.Mode.TOOLS)
    client.load_patch(mode="tool_call")
    assert client.patch is not None

    from litellm import LiteLLM

    # create client w/ litellm
    client = _api_resource.APIResource(provider="litellm")
    assert client is not None
    assert isinstance(client.client, LiteLLM)
    logger.debug(f"api resource inits w/ litellm instance: {client}")

    # patch client
    # use mode: "tool_call" (instructor.Mode.TOOLS)
    client.load_patch(mode="tool_call")
    assert client.patch is not None



if __name__ == "__main__":
    test_api_resource_litellm_resource()
    test_api_resource_init()
    test_api_resource_instructor_patch()