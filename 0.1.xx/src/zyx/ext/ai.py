# zyx ==============================================================================

from ._loader import UtilLazyLoader


class batch_completion(UtilLazyLoader):
    pass


batch_completion.init("litellm.main", "batch_completion_models_all_responses")


class completion(UtilLazyLoader):
    pass


completion.init("zyx.functions.completion", "completion")


class embedding(UtilLazyLoader):
    pass


embedding.init("litellm.main", "embedding")


class Inference(UtilLazyLoader):
    pass


Inference.init("huggingface_hub.inference._client", "InferenceClient")


class ollama_embedding(UtilLazyLoader):
    pass


ollama_embedding.init("ollama", "embeddings")
