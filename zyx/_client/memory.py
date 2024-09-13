from qdrant_client.http import models as rest
from pydantic import BaseModel, Field
from typing import Callable, List, Union, Optional, Literal, Type, Any
import uuid
import json

from .completion import ClientModeParams
from .. import logger


class QdrantNode(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Optional[dict] = None


class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: List[QdrantNode] = Field(default_factory=list)


class Memory:
    def __init__(
        self,
        collection_name: str = "my_collection",
        model_class: Optional[Type[BaseModel]] = None,
        vector_size: int = 1536,
        distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine",
        location: Literal["zyxstore.db", ":memory:"] = "zyxstore.db",
        path : Union[str, Literal["zyxstore.db"]] = None,
        host: str = None,
        port: int = 6333,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: Optional[int] = None,
        embedding_api_key: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        embedding_api_version: Optional[str] = None,
    ):
        """
        Memory class for storing and retrieving data using Qdrant.

        Args:
            collection_name (str): The name of the collection to store data in.
            model_class (Type[BaseModel]): The class of the model to be used for storing and retrieving data. Defaults to None.
            vector_size (int): The size of the vectors to be used for embedding.
            distance (str): The distance metric to be used for similarity search.
            location (str): The location of the Qdrant server. Defaults to ":memory:".
            path (str): The path to the Qdrant server. Defaults to None.
            host (str): The host of the Qdrant server. Defaults to None.
            port (int): The port of the Qdrant server. Defaults to 6333.
            embedding_model (str): The model to be used for embedding. Defaults to "text-embedding-3-small".
            embedding_dimensions (int): The dimensions of the embedding model. Defaults to None.
            embedding_api_key (str): The API key for the embedding model. Defaults to None.
            embedding_api_base (str): The base URL for the embedding model. Defaults to None.
        """
        
        from qdrant_client.http.models import Distance, VectorParams
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.path = path
        self.host = host
        self.port = port
        self.vector_size = vector_size
        self.distance = Distance(distance)
        self.location = location
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_api_key = embedding_api_key
        self.embedding_api_base = embedding_api_base
        self.embedding_api_version = embedding_api_version
        self.model_class = model_class

        if location == ":memory:" or host:
            self.client = QdrantClient(location=location, host=host, port=port)
        else:
            if path:
                self.client = QdrantClient(path=path)
            else:
                self.client = QdrantClient(path=location)

        self._create_collection()

    def _create_collection(self):
        from qdrant_client.http.models import VectorParams

        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.info(
                f"Collection '{self.collection_name}' does not exist. Creating it now."
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=self.distance
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")

    def _get_embedding(self, text: str) -> List[float]:
        from litellm.main import embedding as litellm_embedding

        try:
            response = litellm_embedding(
                model=self.embedding_model,
                input=[text],
                dimensions=self.embedding_dimensions,
                api_key=self.embedding_api_key,
                api_base=self.embedding_api_base,
                api_version=self.embedding_api_version,
            )

            embedding_data = response.get("data", None)
            if (
                embedding_data
                and isinstance(embedding_data, list)
                and len(embedding_data) > 0
            ):
                embedding_vector = embedding_data[0].get("embedding", None)
                if isinstance(embedding_vector, list) and all(
                    isinstance(x, float) for x in embedding_vector
                ):
                    return embedding_vector
                else:
                    raise ValueError(
                        "Invalid embedding format: Expected a list of floats within the 'embedding' key"
                    )
            else:
                raise ValueError("Embedding data is missing or improperly formatted")
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def add(
        self,
        data: Union[str, List[str], BaseModel, List[BaseModel]],
        metadata: Optional[dict] = None,
    ):
        from qdrant_client.http.models import PointStruct

        if isinstance(data, str):
            data = [data]
        elif isinstance(data, BaseModel):
            data = [data]

        points = []
        for item in data:
            try:
                if isinstance(item, BaseModel):
                    text = json.dumps(item.dict())
                    metadata = item.dict()
                else:
                    text = item

                embedding_vector = self._get_embedding(text)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding_vector,
                    payload={
                        "text": text,
                        "metadata": metadata or {},
                        "is_model": isinstance(item, BaseModel),
                    },
                )
                points.append(point)
            except Exception as e:
                logger.error(f"Error processing item: {item}. Error: {e}")

        if points:
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                logger.info(
                    f"Successfully added {len(points)} points to the collection."
                )
            except Exception as e:
                logger.error(f"Error upserting points to collection: {e}")
        else:
            logger.warning("No valid points to add to the collection.")

    def add_docs(self, file_paths: Union[str, List[str]]):
        from pathlib import Path
        from semchunk import chunkerify

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            path = Path(file_path)
            if not path.is_file():
                logger.warning(f"'{file_path}' is not a valid file. Skipping.")
                continue

            try:
                with path.open("r", encoding="utf-8") as file:
                    content = file.read()

                # Initialize chunker
                chunker = chunkerify(self.embedding_model, chunk_size=self.vector_size)
                chunks = chunker(content)

                document_id = str(uuid.uuid4())
                for chunk in chunks:
                    embedding = self._get_embedding(chunk)
                    document = Document(
                        document_id=document_id,
                        text=chunk,
                        metadata={"file_path": str(path)},
                    )

                    point = rest.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "metadata": document.metadata,
                            "document_id": document.document_id,
                        },
                    )

                    self.client.upsert(
                        collection_name=self.collection_name, points=[point]
                    )

                logger.info(f"Successfully processed and added document: {path}")
            except Exception as e:
                logger.error(f"Error processing document {path}: {e}")

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        try:
            query_vector = self._get_embedding(query)
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )

            nodes = []
            for result in search_results:
                payload = result.payload
                if payload.get("is_model", False) and self.model_class:
                    # This is a stored BaseModel instance
                    model_data = json.loads(payload.get("text", "{}"))
                    model_instance = self.model_class(**model_data)
                    node = QdrantNode(
                        id=str(result.id),
                        text=str(model_instance),
                        embedding=query_vector,
                        metadata={"model_data": model_data},
                    )
                elif "document_id" in payload:
                    # This is a document chunk
                    node = QdrantNode(
                        id=str(result.id),
                        text=payload.get("text", ""),
                        embedding=query_vector,
                        metadata={
                            "document_id": payload["document_id"],
                            "file_path": payload["metadata"]["file_path"],
                        },
                    )
                else:
                    # This is a regular node
                    node = QdrantNode(
                        id=str(result.id),
                        text=payload.get("text", ""),
                        embedding=query_vector,
                        metadata=payload.get("metadata", {}),
                    )
                nodes.append(node)
            return SearchResponse(query=query, results=nodes)
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def completion(
        self,
        messages: Union[str, list[dict[str, str]]] = None,
        model: Optional[str] = "gpt-4o-mini",
        top_k: Optional[int] = 5,
        tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
        run_tools: Optional[bool] = True,
        response_model: Optional[BaseModel] = None,
        mode: Optional[ClientModeParams] = "tools",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = 3,
        verbose: Optional[bool] = False,
    ):
        logger.info(f"Initial messages: {messages}")

        # Unwrap the extra array if present
        if (
            isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], list)
        ):
            messages = messages[0]
            logger.info(f"Unwrapped messages from extra array: {messages}")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            if not messages:
                raise ValueError("Messages list is empty")
            # Ensure each item in the list is a dictionary
            messages = [
                {"role": "user", "content": m} if isinstance(m, str) else m
                for m in messages
            ]
            if not all(
                isinstance(m, dict) and "role" in m and "content" in m for m in messages
            ):
                raise ValueError(
                    "Invalid message format. Expected list of dicts with 'role' and 'content' keys"
                )
        else:
            raise ValueError("Invalid message format. Expected str or list of dicts")

        if verbose:
            logger.info(f"Formatted messages: {messages}")

        query = messages[-1].get("content", "") if messages else ""

        try:
            results = self.search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Error during search: {e}")
            results = SearchResponse(query=query)

        results_content = []
        for result in results.results:
            metadata = result.metadata
            metadata_str = ", ".join(
                [f"{key}: {value}" for key, value in metadata.items()]
            )
            results_content.append(f"Text: {result.text}, Metadata: {metadata_str}")

        if verbose:
            logger.info(f"Search results: {results_content}")

        if messages:
            if not any(message.get("role", "") == "system" for message in messages):
                system_message = {
                    "role": "system",
                    "content": f"You have retrieved the following relevant information. Use only if relevant {str(results_content)}",
                }
                messages.insert(0, system_message)
                if verbose:
                    logger.info(f"Inserted system message: {messages}")
            else:
                for message in messages:
                    if message.get("role", "") == "system":
                        message["content"] += (
                            f" You have retrieved the following relevant information. Use only if relevant {str(results_content)}"
                        )
                        if verbose:
                            logger.info(f"Updated system message: {messages}")
                        break

        if verbose:
            logger.info(f"Final messages before ClientParams: {messages}")

        from .completion import CompletionClient

        messages = CompletionClient.format_messages(messages=messages)

        try:
            from .completion import completion

            result = completion(
                messages=messages,
                model=model,
                tools=tools,
                run_tools=run_tools,
                response_model=response_model,
                mode=mode,
                base_url=base_url,
                api_key=api_key,
                organization=organization,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                verbose=verbose,
            )

            if verbose:
                logger.info(f"Completion result: {result}")

            return result
        except Exception as e:
            logger.error(f"Error during completion: {e}")
            logger.exception("Full traceback:")
            raise

    def save(self, path: Optional[str] = None):
        from qdrant_client.qdrant_client import QdrantClient
        from pathlib import Path
        import json

        if self.location != ":memory:":
            logger.warning("This Memory instance is not in-memory. No need to save.")
            return

        if path is None:
            home_dir = Path.home()
            path = home_dir / ".zyx" / "memories"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        collection_path = path / self.collection_name
        collection_path.mkdir(exist_ok=True)

        try:
            # Save collection configuration
            config = {
                "vector_size": self.vector_size,
                "distance": self.distance.name,
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
            }
            with open(collection_path / "config.json", "w") as f:
                json.dump(config, f)

            # Save collection data
            self.client.snapshot(
                collection_name=self.collection_name,
                snapshot_path=str(collection_path / "snapshot"),
            )

            logger.info(f"Successfully saved Memory to {collection_path}")

            # Update the client to use the new location
            self.location = str(collection_path)
            self.client = QdrantClient(path=self.location)

        except Exception as e:
            logger.error(f"Error saving Memory: {e}")
            raise

    def get_model_instances(self, query: str, top_k: int = 5) -> List[Any]:
        if not self.model_class:
            raise ValueError("No model class specified for this Memory instance")

        search_response = self.search(query, top_k)
        model_instances = []

        for result in search_response.results:
            if "model_data" in result.metadata:
                model_instance = self.model_class(**result.metadata["model_data"])
                model_instances.append(model_instance)

        return model_instances


if __name__ == "__main__":
    try:
        qdrant = Memory(collection_name="my_collection", vector_size=1536)
        qdrant.add(["Hello, world!", "How are you?", "What's up?"])
        results = qdrant.search("How are you?")
        for result in results.results:
            print(f"ID: {result.id}, Text: {result.text}, Metadata: {result.metadata}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

    try:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            age: int
            description: str

        pydantic_qdrant = Memory(
            collection_name="pydantic_collection",
            vector_size=1536,
            model_class=TestModel,
            location="zyxstore.db"
        )

        # Add some test models
        test_models = [
            TestModel(name="Alice", age=30, description="Software engineer"),
            TestModel(name="Bob", age=25, description="Data scientist"),
            TestModel(name="Charlie", age=35, description="Product manager"),
        ]
        pydantic_qdrant.add(test_models)

        # Search for models
        search_query = "engineer"
        model_results = pydantic_qdrant.get_model_instances(search_query, top_k=2)

        print(f"\nSearch results for '{search_query}':")
        for model in model_results:
            print(
                f"Name: {model.name}, Age: {model.age}, Description: {model.description}"
            )

        # Test regular search to ensure it still works with model data
        regular_results = pydantic_qdrant.search(search_query, top_k=2)
        print(f"\nRegular search results for '{search_query}':")
        for result in regular_results.results:
            print(f"ID: {result.id}, Text: {result.text}, Metadata: {result.metadata}")

    except Exception as e:
        logger.error(f"Error in Pydantic store test: {e}")
        logger.exception("Full traceback:")

    try:
        import tempfile
        import os

        # Create a temporary directory for the file store
        with tempfile.TemporaryDirectory() as temp_dir:
            file_store_path = os.path.join(temp_dir, "file_store")
            # Create a file store
            file_store = Memory(
                collection_name="file_store", vector_size=1536, location=file_store_path
            )

            # Add some text data
            file_store.add(
                [
                    "This is a test of the file-based store.",
                    "File stores can persist data between sessions.",
                    "Qdrant supports both in-memory and file-based storage.",
                ]
            )

            # Save the file store
            file_store.save()

            print("\nTesting file store:")
            results = file_store.search("file-based storage")
            for result in results.results:
                print(f"ID: {result.id}, Text: {result.text}")

            # Test completions on all three stores
            test_query = "Tell me about data storage"
            stores = [qdrant, pydantic_qdrant, file_store]
            store_names = ["In-memory store", "Pydantic store", "File store"]

            for store, name in zip(stores, store_names):
                print(f"\nTesting completion on {name}:")
                try:
                    completion_result = store.completion(
                        test_query, model="gpt-3.5-turbo"
                    )
                    print(f"Completion result: {completion_result}")
                except Exception as e:
                    print(f"Error during completion: {e}")

    except Exception as e:
        logger.error(f"Error in file store and completion test: {e}")
        logger.exception("Full traceback:")
