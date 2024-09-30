# Multi Document RAG

Multi Document RAG is not as simple as single document QA. `zyx` has decided to let the professionals handle this one, and provides a simple interface
for working with `Document` objects using the **`chromadb`** library, with a simple wrapper interface.

---

## Example

Lets begin by loading the documents we're going to use for this example.

```python
import zyx

links = [
    "https://openreview.net/pdf?id=zAdUB0aCTQ", # AgentBench: Evaluating LLMs as Agents
    "https://openreview.net/pdf?id=z8TW0ttBPp", # MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning
    "https://openreview.net/pdf?id=yoVq2BGQdP", # Achieving Fairness in Multi-Agent MDP Using Reinforcement Learning
    "https://openreview.net/pdf?id=yRrPfKyJQ2", # Conversational Drug Editing Using Retrieval and Domain Feedback
]

documents = zyx.read(links)                     # Handles lists of links/paths as well
```

Lets now investigate the documents we read

```python
for doc in documents:
    print("---")
    print(doc.content[:200])
```

<details>
<summary>Output</summary>
```bash
---
Published as a conference paper at ICLR 2024
AGENT BENCH : EVALUATING LLM S AS AGENTS
Xiao Liu1,*, Hao Yu1,*,†, Hanchen Zhang1,*, Yifan Xu1, Xuanyu Lei1, Hanyu Lai1, Yu Gu2,†,
Hangliang Ding1, Kaiwen
---
Published as a conference paper at ICLR 2024
MATHCODER : S EAMLESS CODE INTEGRATION IN
LLM S FOR ENHANCED MATHEMATICAL REASONING
Ke Wang1,4∗Houxing Ren1∗Aojun Zhou1∗Zimu Lu1∗Sichun Luo3∗
Weikang Shi1∗
---
Published as a conference paper at ICLR 2024
ACHIEVING FAIRNESS IN MULTI -AGENT MDP U SING
REINFORCEMENT LEARNING
Peizhong Ju
Department of ECE
The Ohio State University
Columbus, OH 43210, USA
ju.171
---
Published as a conference paper at ICLR 2024
CONVERSATIONAL DRUG EDITING USING RETRIEVAL
AND DOMAIN FEEDBACK
Shengchao Liu1 *, Jiongxiao Wang2 *, Yijin Yang3, Chengpeng Wang4, Ling Liu5,
Hongyu Guo6,7
```
</details>

---

## Creating a Memory Store

```python
# Initialize an on memory store
store = zyx.Memory()
```

Now lets add our documents to the store

```python
store.add(documents)

# Now we can use the store to search for documents
# One of our papers is about LLM's in the domain of Drug Editing
results = store.search("Drug Editing")
```

---

### LLM Completions in the Store

```python
# We can also wuery our store with an LLM
response = store.completion("How have LLM's been used in the domain of Drug Editing?")

print(response)
```

<details>
<summary>Output</summary>
```bash
ChatCompletion(
    id='chatcmpl-ACvGG7JCm2pCwIZgxNCQa5Iew9HEZ',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content='Large Language Models (LLMs) have been utilized in the domain of drug editing primarily for their capabilities in data analysis,
predictive modeling, and natural language processing. They assist in the identification of potential drug candidates by analyzing vast databases of chemical
compounds and biological data. LLMs can predict the interactions between drugs and biological targets, facilitate the design of novel drug molecules, and
streamline the drug discovery process by automating literature reviews and synthesizing relevant information. Moreover, their ability to generate hypotheses and
simulate molecular interactions aids researchers in optimizing drug formulations and improving efficacy. Overall, LLMs enhance efficiency and innovation in drug
editing and development.',
                refusal=None,
                role='assistant',
                function_call=None,
                tool_calls=None
            )
        )
    ],
    created=1727643412,
    model='gpt-4o-mini-2024-07-18',
    object='chat.completion',
    service_tier=None,
    system_fingerprint='fp_f85bea6784',
    usage=CompletionUsage(completion_tokens=126, prompt_tokens=46, total_tokens=172, completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0))
)
```
</details>

---

## API Reference

::: zyx.resources.stores.memory.Memory
