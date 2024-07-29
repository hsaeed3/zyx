
**zyx**

**A wrapper on top of very nice libraries**
- phidata
- marvin
- instructor
- litellm

</br>

**quick start**

```bash
pip install zyx
```

or with uv

```bash
pip install uv

uv pip install zyx
```

</br>

**@marvin magic functions**

```python
@zyx.function
def auth_keys(amount : int):
    """Generates x amount of authentication keys in an encypted hash format"""

auth_keys(5)
```

```bash
# OUTPUT
'["a3f5e16d", "b4d8a56c", "c9b7e14d", "d1c7b39e", "e6a9f25d"]'
```

**text generation**

```python
# Completions

zyx.completion(
    "Hello, how are you today?",
    model = "ollama/llama3.1"
)
```

```python
class CharacterModel(zyx.BaseModel):
    name : str
    facts : list

response = zyx.instructor_completion(
    "who is harry potter?",
    model = "openai/gpt-4o"
)
```

```python
class ResponseModel(zyx.BaseModel):
    urls : list[str]
    winner : str
    score : str
    
print(zyx.completion("who won the 2024 euro cup final?", 
                          tools = ["web"], 
                          response_model = ResponseModel))
```

```bash
# OUTPUT
{
    "urls": [
        "https://www.cnn.com/2024/07/14/sport/spain-england-euro-2024-final-spt-intl/index.html",
        "https://www.nytimes.com/athletic/live-blogs/england-spain-live-updates-euro-2024-final-score-result/T1jQng4KWoo4/",
        "https://www.espn.com.au/football/story/_/id/40560458/euro-2024-final-england-spain-reaction-analysis-highlights",
        "https://www.usatoday.com/story/sports/soccer/europe/2024/07/14/england-vs-spain-euro-2024-odds-prediction-highlights-result/74399785007/",
        "https://www.youtube.com/watch?v=Ya6eJjmklS8"
    ],
    "winner": "Spain",
    "score": "2-1"
}
```

**nlp**

```python
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Monkey see, monkey do.",
    "Who is the president of the United States?"
]

for text in texts:
    print(zyx.ai.classify(text, 
                          labels = ["monkey", "not_monkey"]))
```

```python
class SubjectModel(zyx.BaseModel):
    subject : str
    action : str
    complexity : float 
    
# Generate Synthetic Data (Marvin)
print(zyx.ai.generate(SubjectModel))

for text in texts:
    # Cast into a model (Marvin)
    print(zyx.ai.cast(text,
                      SubjectModel))

    # Extract into a model, Answering the question (Marvin)
    print(zyx.ai.extract(text,
                         SubjectModel))
```


