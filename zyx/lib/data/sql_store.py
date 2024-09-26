from typing import List, Optional, Union, Any
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pydantic import BaseModel
import uuid
from rank_bm25 import BM25Okapi
import json
from pathlib import Path
from loguru import logger



class Document(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    text: str
    metadata: dict = Field(default_factory=dict)

class SearchResult(BaseModel):
    id: str
    text: str
    metadata: Optional[dict] = None
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult] = Field(default_factory=list)

class Sql:
    def __init__(
        self,
        db_url: str = "sqlite:///store.db",
        model_class: Optional[type] = None,
    ):
        self.engine = create_engine(db_url)
        SQLModel.metadata.create_all(self.engine)
        self.model_class = model_class
        self.bm25 = None
        self.documents = []
        self._load_documents()

    def _load_documents(self):
        with Session(self.engine) as session:
            documents = session.exec(select(Document)).all()
            self.documents = [doc.text for doc in documents]
        self.bm25 = BM25Okapi([doc.split() for doc in self.documents])

    def add(self, data: Union[str, List[str], BaseModel, List[BaseModel]], metadata: Optional[dict] = None):
        with Session(self.engine) as session:
            if isinstance(data, str):
                data = [data]
            elif isinstance(data, BaseModel):
                data = [data]

            for item in data:
                if isinstance(item, BaseModel):
                    text = json.dumps(item.dict())
                    metadata = item.dict()
                else:
                    text = item

                document = Document(text=text, metadata=metadata or {})
                session.add(document)
                self.documents.append(text)

            session.commit()

        self.bm25 = BM25Okapi([doc.split() for doc in self.documents])

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        with Session(self.engine) as session:
            for idx, score in top_n:
                document = session.exec(select(Document).offset(idx).limit(1)).first()
                if document:
                    result = SearchResult(
                        id=document.id,
                        text=document.text,
                        metadata=document.metadata,
                        score=score
                    )
                    results.append(result)

        return SearchResponse(query=query, results=results)

    def get_model_instances(self, query: str, top_k: int = 5) -> List[Any]:
        if not self.model_class:
            raise ValueError("No model class specified for this Store instance")

        search_response = self.search(query, top_k)
        model_instances = []

        for result in search_response.results:
            if isinstance(result.metadata, dict):
                model_instance = self.model_class(**result.metadata)
                model_instances.append(model_instance)

        return model_instances

    def save(self, path: Optional[str] = None):
        if path is None:
            home_dir = Path.home()
            path = home_dir / ".zyx" / "stores"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        db_path = path / "store.db"
        new_engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(new_engine)

        with Session(self.engine) as source_session, Session(new_engine) as target_session:
            documents = source_session.exec(select(Document)).all()
            for document in documents:
                new_document = Document(id=document.id, text=document.text, metadata=document.metadata)
                target_session.add(new_document)
            target_session.commit()

        logger.info(f"Successfully saved Store to {db_path}")

    @classmethod
    def load(cls, path: str):
        db_path = Path(path) / "store.db"
        if not db_path.exists():
            raise FileNotFoundError(f"No database found at {db_path}")

        store = cls(db_url=f"sqlite:///{db_path}")
        store._load_documents()
        return store

if __name__ == "__main__":
    # Test the Store class
    store = Sql()

    # Add some test data
    store.add(["Hello, world!", "How are you?", "What's up?"])

    # Test search
    results = store.search("How are you?")
    print("Search results:")
    for result in results.results:
        print(f"ID: {result.id}, Text: {result.text}, Score: {result.score}")

    # Test with Pydantic models
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str
        age: int
        description: str

    pydantic_store = Sql(model_class=TestModel)

    # Add some test models
    test_models = [
        TestModel(name="Alice", age=30, description="Software engineer"),
        TestModel(name="Bob", age=25, description="Data scientist"),
        TestModel(name="Charlie", age=35, description="Product manager"),
    ]
    pydantic_store.add(test_models)

    # Search for models
    search_query = "engineer"
    model_results = pydantic_store.get_model_instances(search_query, top_k=2)

    print(f"\nSearch results for '{search_query}':")
    for model in model_results:
        print(f"Name: {model.name}, Age: {model.age}, Description: {model.description}")

    # Test save and load
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the store
        store.save(temp_dir)

        # Load the store
        loaded_store = Sql.load(temp_dir)

        # Test search on loaded store
        results = loaded_store.search("How are you?")
        print("\nSearch results from loaded store:")
        for result in results.results:
            print(f"ID: {result.id}, Text: {result.text}, Score: {result.score}")