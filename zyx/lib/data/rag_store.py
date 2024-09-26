from typing import List, Optional, Union, Any
from pydantic import BaseModel
from pathlib import Path

# Import the VectorStore and Store classes
from .vector_store import VectorStore
from .sql_store import Sql
from loguru import logger
from ..types.document import Document  # Import Document from document.py


class CombinedSearchResult(BaseModel):
    id: str
    text: str
    metadata: Optional[dict] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    combined_score: float


class CombinedSearchResponse(BaseModel):
    query: str
    vector_results: List[Any]
    bm25_results: List[Any]
    combined_results: List[CombinedSearchResult]


class Rag:
    def __init__(
        self,
        collection_name: str = "my_collection",
        vector_size: int = 1536,
        db_url: str = "sqlite:///store.db",
        model_class: Optional[type] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.vector_store = VectorStore(
            collection_name=collection_name,
            vector_size=vector_size,
            model_class=model_class,
            embedding_model=embedding_model,
        )
        self.bm25_store = Sql(db_url=db_url, model_class=model_class)
        self.model_class = model_class

    def add(
        self,
        data: Union[str, List[str], Document, List[Document]],
        metadata: Optional[dict] = None,
    ):
        self.vector_store.add(data, metadata)
        self.bm25_store.add(data, metadata)

    def _combine_results(self, vector_results, bm25_results, top_k: int = 5):
        combined_results = {}

        for result in vector_results.results:
            combined_results[result.id] = {
                "id": result.id,
                "text": result.text,
                "metadata": result.metadata,
                "vector_score": 1.0,  # Normalize vector scores
                "bm25_score": 0.0,
            }

        max_bm25_score = (
            max(result.score for result in bm25_results.results)
            if bm25_results.results
            else 1.0
        )
        for result in bm25_results.results:
            if result.id in combined_results:
                combined_results[result.id]["bm25_score"] = (
                    result.score / max_bm25_score
                )
            else:
                combined_results[result.id] = {
                    "id": result.id,
                    "text": result.text,
                    "metadata": result.metadata,
                    "vector_score": 0.0,
                    "bm25_score": result.score / max_bm25_score,
                }

        for result in combined_results.values():
            result["combined_score"] = (
                result["vector_score"] + result["bm25_score"]
            ) / 2

        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["combined_score"], reverse=True
        )
        return [CombinedSearchResult(**result) for result in sorted_results[:top_k]]

    def search(self, query: str, top_k: int = 5) -> CombinedSearchResponse:
        vector_results = self.vector_store.search(query, top_k=top_k)
        bm25_results = self.bm25_store.search(query, top_k=top_k)
        combined_results = self._combine_results(vector_results, bm25_results, top_k)

        return CombinedSearchResponse(
            query=query,
            vector_results=vector_results.results,
            bm25_results=bm25_results.results,
            combined_results=combined_results,
        )

    def get_model_instances(self, query: str, top_k: int = 5) -> List[Any]:
        if not self.model_class:
            raise ValueError("No model class specified for this RagStore instance")

        search_response = self.search(query, top_k)
        model_instances = []

        for result in search_response.combined_results:
            if isinstance(result.metadata, dict):
                model_instance = self.model_class(**result.metadata)
                model_instances.append(model_instance)

        return model_instances

    def completion(self, messages: Union[str, list[dict[str, str]]], **kwargs):
        # Perform search using the last message
        if isinstance(messages, str):
            query = messages
        elif isinstance(messages, list):
            query = messages[-1]["content"]
        else:
            raise ValueError("Invalid message format")

        search_results = self.search(query)

        # Prepare context from search results
        context = "\n".join(
            [
                f"Text: {result.text}, Score: {result.combined_score}"
                for result in search_results.combined_results
            ]
        )

        # Add context to the system message
        if isinstance(messages, str):
            messages = [
                {
                    "role": "system",
                    "content": f"You have the following context:\n{context}",
                },
                {"role": "user", "content": messages},
            ]
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"You have the following context:\n{context}",
                },
            )

        return self.vector_store.completion(messages=messages, **kwargs)

    def save(self, path: Optional[str] = None):
        if path is None:
            home_dir = Path.home()
            path = Path(home_dir) / ".zyx" / "rag_stores"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        vector_store_path = path / "vector_store"
        bm25_store_path = path / "bm25_store"

        self.vector_store.save(str(vector_store_path))
        self.bm25_store.save(str(bm25_store_path))

        logger.info(f"Successfully saved RagStore to {path}")

    @classmethod
    def load(cls, path: str, model_class: Optional[type] = None):
        path = Path(path)
        vector_store_path = path / "vector_store"
        bm25_store_path = path / "bm25_store"

        if not vector_store_path.exists() or not bm25_store_path.exists():
            raise FileNotFoundError(f"Store data not found in {path}")

        rag_store = cls(model_class=model_class)
        rag_store.vector_store = VectorStore.load(str(vector_store_path))
        rag_store.bm25_store = Sql.load(str(bm25_store_path))

        return rag_store


if __name__ == "__main__":
    # Test the RagStore class
    rag_store = Rag()

    # Add some test data
    rag_store.add(["Hello, world!", "How are you?", "What's up?"])

    # Test search
    results = rag_store.search("How are you?")
    print("Search results:")
    for result in results.combined_results:
        print(
            f"ID: {result.id}, Text: {result.text}, Combined Score: {result.combined_score}"
        )

    # Test with Pydantic models
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str
        age: int
        description: str

    pydantic_rag_store = Rag(model_class=TestModel)

    # Add some test models
    test_models = [
        TestModel(name="Alice", age=30, description="Software engineer"),
        TestModel(name="Bob", age=25, description="Data scientist"),
        TestModel(name="Charlie", age=35, description="Product manager"),
    ]
    pydantic_rag_store.add(test_models)

    # Search for models
    search_query = "engineer"
    model_results = pydantic_rag_store.get_model_instances(search_query, top_k=2)

    print(f"\nSearch results for '{search_query}':")
    for model in model_results:
        print(f"Name: {model.name}, Age: {model.age}, Description: {model.description}")

    # Test completion
    completion_result = rag_store.completion(
        "Tell me about data storage", model="gpt-3.5-turbo"
    )
    print("\nCompletion result:")
    print(completion_result)

    # Test save and load
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the store
        rag_store.save(temp_dir)

        # Load the store
        loaded_rag_store = Rag.load(temp_dir)

        # Test search on loaded store
        results = loaded_rag_store.search("How are you?")
        print("\nSearch results from loaded store:")
        for result in results.combined_results:
            print(
                f"ID: {result.id}, Text: {result.text}, Combined Score: {result.combined_score}"
            )
