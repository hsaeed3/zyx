# LLM as a Judge

The `judge()` function is a multi-use tool, that can be used for:

- **Hallucination Detection**
- **Schema Validation**
- **Multi-Response Accuracy**
- **Guardrails Enforcement**
- **Response Regeneration**

---

## Using the `judge()` function

### Fact Checking

For most quick fact-checking tasks, the judge function can be utilized without any verifier schema or instructions.

```python
import zyx

fact = "The capital of France is India"

zyx.judge(
    fact,
    process = "fact_check",
    model = "gpt-4o-mini"
)
```

```bash
FactCheckResult(
    is_accurate=False,
    explanation='The statement that the capital of France is India is incorrect. The capital of France is Paris, not India. India is a country
in South Asia, and it has its own capital, which is New Delhi.',
    confidence=1.0
)
```

---

### Accuracy Judgement

You can use the `judge()` function to compare multiple responses to a given prompt and determine which one is the most accurate, helpful, and relevant.

```python
import zyx

prompt = "Explain the theory of relativity"
responses = [
    "The theory of relativity describes how space and time are interconnected.",
    "Einstein's theory states that E=mc^2, which relates energy and mass."
]

zyx.judge(
    prompt,
    responses=responses,
    process="accuracy"
)
```

```bash
JudgmentResult(
    explanation='Response 1 provides a broader understanding of the theory of relativity by mentioning the interconnection of space and time,
which is fundamental to both the special and general theories of relativity. Response 2 focuses on the famous equation E=mc^2, which is a key
result of the theory but does not provide a comprehensive explanation of the overall theory. Therefore, while both responses are accurate,
Response 1 is more helpful and relevant as it captures a core aspect of the theory.',
    verdict='Response 1 is the most accurate, helpful, and relevant.'
)
```

---

### Schema Validation

The `judge()` function can also be used to validate responses against a predefined schema or set of criteria.

```python
import zyx

prompt = "Describe the water cycle"
response = "Water evaporates, forms clouds, and then falls as rain."
schema = "The response should include: 1) Evaporation, 2) Condensation, 3) Precipitation, 4) Collection"

result = zyx.judge(
    prompt,
    responses=response,
    process="validate",
    schema=schema
)

print(result)
```

```bash
ValidationResult(
    is_valid=False,
    explanation='The response does not include all required components of the water cycle as outlined in the schema. Specifically, it mentions
evaporation and precipitation, but it fails to mention condensation and collection.'
)
```

---

## Response Regeneration

An important functionality of this module is the ability to regenerate a correct response if the original one was determined to be inaccurate or incomplete. This can be useful for generating high-quality responses for a given prompt based on the schema or instructions provided.

```python hl_lines="13"
import zyx

prompt = "Explain photosynthesis"
responses = [
    "Photosynthesis is how plants make food.",
    "Plants use sunlight to convert CO2 and water into glucose and oxygen."
]

regenerated_response = zyx.judge(
    prompt,
    responses=responses,
    process="accuracy",
    regenerate=True,
    verbose=True
)

print(regenerated_response)
```

```bash hl_lines="3"
[09/29/24 00:42:45] INFO     judge - judge - Judging responses for prompt: Explain photosynthesis
...
[09/29/24 00:42:47] WARNING  judge - judge - Response is not accurate. Regenerating response.
...
RegeneratedResponse(
    response='Photosynthesis is a biochemical process used by plants...'
)
```

---

## Guardrails

The final big functionality of the `judge()` function is the ability to enforce guardrails on the responses generated. This can help ensure that the responses are accurate, relevant, and appropriate for the given prompt.
**If a response violates guardrails, it will always be regenerated.**

```python
import zyx

prompt = "Describe the benefits of exercise"
responses = ["Exercise helps you lose weight and build muscle."]
guardrails = [
    "Ensure the response mentions mental health benefits.",
    "Include at least three distinct benefits of exercise.",
    "Avoid focusing solely on physical appearance."
]

result = zyx.judge(
    prompt,
    responses=responses,
    process="accuracy",
    guardrails=guardrails,
    verbose=True
)

print(result)
```

```bash hl_lines="3"
[09/29/24 00:50:30] INFO     judge - judge - Judging responses for prompt: Describe the benefits of exercise
...
[09/29/24 00:50:33] WARNING  judge - judge - Response violates guardrails. Regenerating response.
...
RegeneratedResponse(
    response="Exercise offers a multitude of benefits that extend beyond..."
)
```

---

## API Reference

::: zyx.resources.completions.agents.judge.judge
