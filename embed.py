from phi.embedder.ollama import OllamaEmbedder

embeddier = OllamaEmbedder(model="mxbai-embed-large")
print(embeddier.get_embedding("hello world"))
