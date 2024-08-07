# zyx ==============================================================================

__all__ = ["embeddings"]

from typing import List, Literal, Optional, Union

EmbeddingsProviders = ["google/", "openai/", "ollama/"]


def embeddings(
    inputs: Union[list[str], str],
    model: Optional[str] = "openai/text-embedding-ada-002",
    dimensions: Optional[int] = None,
    host: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    encoding_format: Literal["float", "base64"] = "float",
    verbose: Optional[bool] = False,
    *args,
    **kwargs,
):
    """Utilize an external service to retrieve embeddings for the inputs text.

    Args:
        inputs (Union[list[str], str]): The inputs text to embed.
        model (Optional[str], optional): The model to use for embeddings. Defaults to "openai/text-embedding-ada-002".
        dimensions (Optional[int], optional): The number of dimensions for the embeddings. Defaults to None.
        host (Optional[str], optional): The host for the Ollama service. Defaults to None.
        api_key (Optional[str], optional): The API key for the service. Defaults to None.
        base_url (Optional[str], optional): The base URL for the service. Defaults to None.
        organization (Optional[str], optional): The organization for the service. Defaults to None.
        encoding_format (Literal["float", "base64"], optional): The encoding format for the embeddings. Defaults to "float".
        verbose (Optional[bool], optional): Whether to log information. Defaults to False.
    """
    if not inputs:
        raise ValueError("Inputs text is required.")
    if verbose is True:
        from zyx.core import logger

    if not any([model.startswith(provider) for provider in EmbeddingsProviders]):
        from phi.embedder.openai import OpenAIEmbedder

        try:
            embedder = OpenAIEmbedder(
                dimensions=dimensions,
                model=model,
                encoding_format=encoding_format,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
            )

            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize OpenAI Embedder: {e}")
            raise e
    elif model.startswith("openai/"):
        from phi.embedder.openai import OpenAIEmbedder

        model = model[7:]
        try:
            embedder = OpenAIEmbedder(
                dimensions=dimensions if dimensions is not None else 1536,
                model=model,
                encoding_format=encoding_format,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
            )
            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize OpenAI Embedder: {e}")
            raise e
    elif model.startswith("google/"):
        model = model[8:]
        from phi.embedder.google import GeminiEmbedder

        try:
            embedder = GeminiEmbedder(
                dimensions=dimensions if dimensions is not None else 1536,
                model=model,
                api_key=api_key,
            )
            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize Google Embedder: {e}")
            raise e
    elif model.startswith("ollama/"):
        model = model[8:]
        from phi.embedder.ollama import OllamaEmbedder

        try:
            embedder = OllamaEmbedder(
                dimensions=dimensions if dimensions is not None else 4096,
                model=model,
                api_key=api_key,
                host=host,
            )
            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize Ollama Embedder: {e}")
            raise e
