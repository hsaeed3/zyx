# **zyx**

> **zyx 0.3.00** - the first '*release*' of the library is out! </br>
> All CrewAI & other obstructions have been removed, and the library is back to being lightweight.

<code>zyx</code> is a **hyper-fast**, **fun**, & **ease-of-use** focused Python library for using LLMs. </br>
It was created on top of <code>Instructor</code> and <code>LiteLLM</code>, and focuses to provide an abstraction free framework. </br>
The library uses methods such as lazy-loading to provide a single import for all its features. This library is not meant to be used as a production-ready solution, but rather as a tool to quickly & easily experiment with LLMs. 

Some of the key features of <code>zyx</code> include:

- **Universal Completion Client** : A singular function that handles *all LiteLLM compatible models*, *Pydantic structured outputs*, *tool calling & execution*, *prompt optimization*, *streaming* & *vision support*.
- **A Large Collection of LLM Powered Functions** : This library is inspired by <code>MarvinAI</code>, and it's quick LLM function style framework and has built upon it vastly.
- **Easy to Use Memory (Rag)** : A <code>Qdrant</code> wrapper, built to support easy *store creation*, *text/document/pydantic model data insertion*, *universal embedding provider support*, *LLM completions for RAG* & more.
- **Multimodel Generations** : Supports generations for **images**, **audio** & **speech** transcription.
- **Functional / Easy Access Terminal Client** : A terminal client built using <code>textual</code> to allow for easy access to <code>zyx</code> features.
- ***New* Experimental Conversational Multi-Agent Framework** : Built from the ground up using <code>Instructor</code>, the agentic framework provides a solution towards conversationally state managed agents, with *task creation*, *custom tool use*, *artifact creation* & more.

## **Getting Started**

### Installation

```bash
pip install zyx
```

### Generating Completions 

<details closed>
<summary>Open</summary>
<br>

The primary module of zyx, is the universal <code>.completion()</code> function. This module is an extensive wrapper around the <code>litellm .completion()</code> function, as well as the Instructor library. </br>

The <code>.completion()</code> function is capable of

- **Generations with any LiteLLM compatible model**
    - Ollama, OpenAI, Anthropic, Groq, Mistral, and more!
- **Direct Instructor Pydantc structured outputs**
- **Tool calling & execution support. (Get a tool interpretation with one function)**
    - zyx provides a few prebuilt tools out of the box
    - Can take in a list of **Python functions**, **OpenAI dictionaries**, or **Pydantic models** as tools!
    - Automatic tool execution if a tool is called through the <code>run_tools</code> parameter
- **Streaming**
- **New** Vision support 
    - Pass in a list of urls
    - Currently uses multi shot prompting if a response model or tools were also passed.
-  **New** Prompt optimization 
    - Creates or optimizes a task tuned system prompt using either the *COSTAR* or *TIDD-EC* frameworks automatically.

## Standard Completion

```python
# Simplest Way to Generate
# Defaults to "gpt-4o-mini" if no model is provided
from zyx import completion

response = completion("Hi, how are you?")

# Returns a standard OpenAI style response object
print(response.choices[0].message.content)
```

```bash
# OUTPUT
Hello! I'm just a program, but I'm here and ready to help you. How can I assist you today?
```

## Instructor Output

```python
import zyx
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# We can pass in a system prompt to change the behavior of the assistant
response = zyx.completion(
    "Create a mock person",
    response_model = Person
)

# Lets print the full response object
print(response)

print(f"Person Name: '{response.name}', Person Age: '{response.age}'")
```

```bash
# OUTPUT
Person(name='John Doe', age=30)
Person Name: 'John Doe', Person Age: '30'
```

## Tool Calling & Execution

```python
# Lets return a tool call
import zyx

# Lets use the prebuilt web search tool!
response = zyx.completion(
    "Who won the 2024 Euro Cup Final?",
    tools = [zyx.tools.web_search],
    run_tools = True # Set to true to execute tool calls
)

print(response.choices[0].message.content)
```

```bash
# OUTPUT
Spain won the 2024 Euro Cup Final, defeating England 2-1. The decisive goal was scored by substitute Mikel 
Oyarzabal in the 86th minute. This victory marked Spain's fourth European championship title. You can find more 
details about the match (https://en.wikipedia.org/wiki/UEFA_Euro_2024_Final).
```

</details>

### LLM Powered Functions | Code Generators

<details closed>
<summary>Open</summary>
<br>

</details>

### LLM Powered Functions | NLP

<details closed>
<summary>Open</summary>
<br>

</details>

### LLM Powered Functions | Prompt Optimization

<details closed>
<summary>Open</summary>
<br>

</details>

### LLM Powered Functions | Reasoning (Research Paper Implmenetations)

<details closed>
<summary>Open</summary>
<br>

</details>


