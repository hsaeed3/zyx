# **Memory (RAG)**

The <samp>zyx</samp> library brings an easier way to interact with vector databases & implement RAG pipelines. Using the <code>Qdrant-Client</code> library, we can easily store and query data. The <code>completion</code> client has also been built into the Memory module to allow for LLM (RAG) completions.

## **Creating a Store**

Lets create a store to store and query data.

```python
import zyx
from pydantic import BaseModel


class Data(BaseModel):
    memory: str
    date: str


store = zyx.Memory(
    collection_name = "store",
    location = ":memory:",
    model_class = Data,  # Set the Pydantic model class to use.(Optional)
)
```

```bash
# OUTPUT
2024-09-08 00:33:13.825 | INFO     | zyx._client.memory:_create_collection:73 - Collection 'base_store' does not exist. Creating it now.
2024-09-08 00:33:13.826 | INFO     | zyx._client.memory:_create_collection:82 - Collection 'base_store' created successfully.
2024-09-08 00:33:13.827 | INFO     | zyx._client.memory:_create_collection:73 - Collection 'pydantic_store' does not exist. Creating it now.
2024-09-08 00:33:13.827 | INFO     | zyx._client.memory:_create_collection:82 - Collection 'pydantic_store' created successfully.
```

## **Adding Data**

We can add data to the store using the <code>add</code> method.

**Lets generate some mock data to add to the store.**

```python
data = zyx.generate(Data, n=10)

print(data)

# Add it to the store
store.add(pydantic_data)
store.add(
    Data(memory="My bestie and I went to the park yesterday", date="2023-08-01")
)
```

```bash
[
    Data(memory='Remember to buy groceries', date='2023-10-01'),
    Data(memory="Doctor's appointment at 3 PM", date='2023-10-02'),
    Data(memory='Meeting with the project team', date='2023-10-03'),
    Data(memory='Call mom', date='2023-10-04'),
    Data(memory='Finish the report by Friday', date='2023-10-05'),
    Data(memory='Plan weekend trip', date='2023-10-06'),
    Data(memory='Attend yoga class', date='2023-10-07'),
    Data(memory='Submit tax documents', date='2023-10-08'),
    Data(memory='Grocery shopping list', date='2023-10-09'),
    Data(memory='Prepare for the presentation', date='2023-10-10')
]

2024-09-08 00:33:23.230 | INFO     | zyx._client.memory:add:156 - Successfully added 10 points to the collection.
2024-09-08 00:33:23.364 | INFO     | zyx._client.memory:add:156 - Successfully added 1 points to the collection.
```

### **Adding Documents**

Document support is also easily available.

```python
store.add_docs(["doc.txt", "doc2.txt"])
```

## **Querying Data**

We can query the store using the <code>query</code> method.

```python
results = store.search("Did i do anything with my best friend?")

print(results.query)
for result in results.results:
    print(result.text)
```

```bash
# OUTPUT
Did i do anything with my best friend?
memory='My bestie and I went to the park yesterday' date='2023-08-01'
memory='Call mom' date='2023-10-04'
memory='Plan weekend trip' date='2023-10-06'
memory='Meeting with the project team' date='2023-10-03'
memory='Attend yoga class' date='2023-10-07'
```

## **RAG Completion**

We can use the <code>completion</code> client to perform RAG completions.

```python
response = store.completion("Did i do anything with my best friend?")

print(response.choices[0].message.content)
```

```bash
2024-09-08 00:34:39.951 | INFO     | zyx._client.memory:completion:277 - Initial messages: Did i do anything with my best friend?
Yes, according to your memory stored under ID: 80446fe4-acd5-4235-8552-65c01531f84f, you and your "bestie" (best 
friend) went to the park yesterday.
```

## **API Reference**

## **VectorStore (Memory)**

::: zyx.lib.data.vector_store.VectorStore

## **SqlStore**

::: zyx.lib.data.sql_store.SqlStore

## **RagStore**

::: zyx.lib.data.rag_store.RagStore

