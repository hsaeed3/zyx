# zyx

`zyx` is a fundamentally different (and much more fun may I add), way to interact with language and
embedding models to define and iterate through workflows, agents, and more.

## Installation

```bash
pip install zyx
```

## Introduction

The foundational idea of the framework is to provide an interface that standardizes on the OpenAI
Chat Completions API format and rather than focus on the explcit definition of agents, provide a
single, incredibly composable interface: `zyx.run()` and `zyx.arun()`.