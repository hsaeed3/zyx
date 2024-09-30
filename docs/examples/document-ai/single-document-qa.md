# Single Document QA

The simplest form of `RAG` or Retrieval-Augmented Generation is single document QA. `zyx` provides a simple interface for interacting
with documents for LLM interpretation.

---

## Example - The Large Language Monkeys Paper

Lets begin by loading the Large Language Monkeys paper from ARXIV, using the `read()` function. `read()` is able to parse most
document formats, from both local files & file URLs.

> PDF Image & Table extraction is currently in development.

```python
import zyx

# Load the Large Language Monkeys paper
paper = zyx.read("https://arxiv.org/pdf/2407.21787")
```

<br/>

The read function returns a `Document` object by default, lets inspect this document now.

```python
print("Metadata:", paper.metadata)
print("Content:", paper.content[:200])
```

```bash
Metadata:
    {'file_name': '2407.21787', 'file_type': 'application/pdf', 'file_size': 955583}

Content: Large Language Monkeys: Scaling Inference Compute
    with Repeated Sampling
    Bradley Brown∗†‡, Jordan Juravsky∗†, Ryan Ehrlich∗†, Ronald Clark‡, Quoc V. Le§,
    Christopher R´ e†, and Azalia Mirhoseini†§
    †De
```

<br/>

The `Document` object can now easily be queried, achieving our goal of single document QA.

```python
result = document.query("What is this paper about?", model = "gpt-4o-mini")
```

<details>
  <summary>Output</summary>
  ```bash
  ChatCompletion(
      id='chatcmpl-ACk3yliGJKL5ueVazlySEHB8peRSa',
      choices=[
          Choice(
              finish_reason='stop',
              index=0,
              logprobs=None,
              message=ChatCompletionMessage(
                  content='The paper titled "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling" investigates the effectiveness of using
  multiple samples during inference to improve the performance of large language models (LLMs) in solving various tasks. Here are the main points the paper
  covers:\n\n1. **Scaling Inference Compute**: While significant advances in LLM capabilities have been achieved through training larger models, the authors argue
  that inference can also be improved by increasing the number of samples generated for solving a problem, rather than limiting it to a single attempt.\n\n2. **Key
  Parameters**: The research focuses on two primary factors:\n   - **Coverage**: The proportion of problems solved by at least one of the generated samples.\n   -
  **Precision**: The ability to identify correct solutions from multiple generated samples.\n\n3. **Experimental Results**: The authors conducted experiments across
  multiple tasks, including coding and formal proofs, and demonstrated that increasing the number of samples can substantially improve coverage. For instance, in
  coding challenges, the coverage improved from 15.9% to 56% when increasing the number of samples from one to 250, surpassing state-of-the-art models that use
  single attempts.\n\n4. **Cost-Effectiveness**: The paper also finds that using repeated sampling with less expensive models can sometimes be more cost-effective
  than using fewer samples from more powerful models.\n\n5. **Inference Scaling Laws**: The relationship between sample size and coverage appears to follow a
  log-linear trend, suggesting that there are scaling laws for inference similar to those observed in training.\n\n6. **Challenges**: While repeated sampling shows
  promise, the authors note that effective mechanisms for identifying correct solutions from many samples remain a challenge, particularly in domains without
  automatic verification methods.\n\n7. **Future Directions**: Suggestions for further research include improving sample verification methods, exploring multi-turn
  interactions for feedback, and leveraging previous attempts in generating new samples.\n\nOverall, the paper emphasizes the potential of repeated sampling as a
  strategy to enhance the problem-solving capabilities of LLMs during inference and highlights both its advantages and areas for future exploration.',
                  refusal=None,
                  role='assistant',
                  function_call=None,
                  tool_calls=None
              )
          )
      ],
      created=1727600366,
      model='gpt-4o-mini-2024-07-18',
      object='chat.completion',
      service_tier=None,
      system_fingerprint='fp_f85bea6784',
      usage=CompletionUsage(completion_tokens=417, prompt_tokens=20407, total_tokens=20824, completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0))
  )
  ```
</details>




