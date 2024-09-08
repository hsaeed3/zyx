# **zyx**


``` bash
pip install zyx --upgrade

zyx
```

</br>

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

### Links

- [Getting Started](examples/getting-started.md)

