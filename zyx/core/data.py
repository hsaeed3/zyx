from peewee import SqliteDatabase, Model, CharField, TextField
from typing import Optional, Union
import os

sqlite_db = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = sqlite_db


class Document(BaseModel):
    id = CharField(primary_key=True)
    text = TextField()
    metadata = TextField(null=True)


def embeddings(
    text: Union[str, list[str]],
    model: Optional[str] = "openai/text-embedding-3-small",
    dimensions: Optional[int] = 1536,
    api_key: Optional[str] = None,
    verbose: Optional[bool] = False,
):
    if isinstance(text, list):
        return [embeddings(t, model, dimensions, api_key) for t in text]
    else:
        if model.startswith("openai/") or model.startswith("text-embedding"):
            from phi.embedder.openai import OpenAIEmbedder

            model = model.replace("openai/", "")
            embedder = OpenAIEmbedder(model=model, dimensions=dimensions)
            if verbose:
                print(
                    f"Initialized [bold green]{model}[/bold green] through the [bold blue]OpenAI[/bold blue] Client."
                )
            return embedder.get_embedding(text)
        elif model.startswith("ollama/"):
            from phi.embedder.ollama import OllamaEmbedder

            model = model.replace("ollama/", "")
            embedder = OllamaEmbedder(model=model, dimensions=dimensions)
            if verbose:
                print(
                    f"Initialized [bold green]{model}[/bold green] through the [bold blue]Ollama[/bold blue] Client."
                )
            return embedder.get_embedding(text)
        elif model.startswith("mistral/"):
            from phi.embedder.mistral import MistralEmbedder

            model = model.replace("mistral/", "")
            embedder = MistralEmbedder(dimensions=dimensions, model=model)
            if verbose:
                print(
                    f"Initialized [bold green]{model}[/bold green] through the [bold blue]Mistral[/bold blue] Client."
                )
            return embedder.get_embedding(text)
        elif model.startswith("google/"):
            from phi.embedder.google import GeminiEmbedder

            model = model.replace("google/", "")
            embedder = GeminiEmbedder(model=model, dimensions=dimensions)
            if verbose:
                print(
                    f"Initialized [bold green]{model}[/bold green] through the [bold blue]Google[/bold blue] Client."
                )
            return embedder.get_embedding(text)


class Database:
    def __init__(
        self,
        location: Optional[str] = ":memory:",
        dimensions: Optional[int] = 1536,
        embedding_model: Optional[str] = "openai/text-embedding-3-small",
    ):
        from qdrant_client import QdrantClient

        self.embedder = self._setup_embedder(embedding_model, dimensions)
        self.location = location
        self.vector_size = dimensions

        if location == ":memory:":
            sqlite_db.init(":memory:")
            self.qdrant_client = QdrantClient(":memory:")
        else:
            sqlite_path = os.path.join(location, "unified.db")
            qdrant_path = os.path.join(location, "qdrant_data")
            sqlite_db.init(sqlite_path)
            self.qdrant_client = QdrantClient(path=qdrant_path)

        self._initialize_databases()

    def _setup_embedder(self, model: str, dimensions: int):
        return lambda text: embeddings(text, model, dimensions)

    def _initialize_databases(self):
        from qdrant_client.http import models as rest_models

        sqlite_db.connect()
        sqlite_db.create_tables([Document])

        if not self.qdrant_client.get_collections().collections:
            self.qdrant_client.create_collection(
                collection_name="documents",
                vectors_config=rest_models.VectorParams(
                    size=self.vector_size, distance=rest_models.Distance.COSINE
                ),
            )

    def add_document(self, text: Union[str, list[str]], metadata=None):
        from qdrant_client.http import models as rest_models
        from uuid import uuid4

        if isinstance(text, str):
            text = [text]

        embeddings = self.embedder(text)
        doc_ids = []

        for t, e in zip(text, embeddings):
            doc_id = str(uuid4())
            Document.create(id=doc_id, text=t, metadata=str(metadata))

            self.qdrant_client.upsert(
                collection_name="documents",
                points=[
                    rest_models.PointStruct(
                        id=doc_id, vector=e, payload={"text": t, "metadata": metadata}
                    )
                ],
            )
            doc_ids.append(doc_id)

        return doc_ids

    def search(self, query: str, max_results: Optional[int] = 5):
        embedding = self.embedder(query)

        vector_results = self.qdrant_client.search(
            collection_name="documents", query_vector=embedding, limit=max_results
        )
        text_results = (
            Document.select().where(Document.text.contains(query)).limit(max_results)
        )

        combined_results = {
            "vector_results": [
                {
                    "id": result.id,
                    "text": result.payload["text"],
                    "metadata": result.payload["metadata"],
                    "score": result.score,
                }
                for result in vector_results
            ],
            "text_results": [
                {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
                for doc in text_results
            ],
        }
        return combined_results

    def update_document(
        self, doc_id: str, text: Optional[str] = None, metadata: Optional[dict] = None
    ):
        from qdrant_client.http import models as rest_models

        update_data = {}
        if text is not None:
            update_data[Document.text] = text
        if metadata is not None:
            update_data[Document.metadata] = str(metadata)

        if update_data:
            updated_rows = (
                Document.update(update_data).where(Document.id == doc_id).execute()
            )
            if updated_rows == 0:
                return False

        if text is not None or metadata is not None:
            current_doc = Document.get_or_none(Document.id == doc_id)
            if current_doc:
                new_text = text if text is not None else current_doc.text
                new_metadata = (
                    metadata if metadata is not None else eval(current_doc.metadata)
                )

                new_embedding = self.embedder(new_text)

                self.qdrant_client.upsert(
                    collection_name="documents",
                    points=[
                        rest_models.PointStruct(
                            id=doc_id,
                            vector=new_embedding,
                            payload={"text": new_text, "metadata": new_metadata},
                        )
                    ],
                )
            else:
                return False

        return True

    def remove_document(self, doc_id: str):
        from qdrant_client.http import models as rest_models

        deleted_rows = Document.delete().where(Document.id == doc_id).execute()

        self.qdrant_client.delete(
            collection_name="documents",
            points_selector=rest_models.PointIdsList(points=[doc_id]),
        )

        return deleted_rows > 0


def database(
    location: Optional[str] = ":memory:",
    dimensions: Optional[int] = 1536,
    embedding_model: Optional[str] = "openai/text-embedding-3-small",
):
    """
    Creates a new database instance.

    Parameters:
        location (str): The location of the database. If set to ":memory:", the database will be stored in memory.
        dimensions (int): The dimensions of the embeddings.
        embedding_model (str): The embedding model to use.
    """
    return Database(location, dimensions, embedding_model)
