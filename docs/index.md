---
icon: lucide/book
title: ""
hide:
  - title
---

<span class="page-index"></span>

![ZYX Hero](./assets/zyx-light.png){ align=center .hero-light }
![ZYX Hero](./assets/zyx-dark.png){ align=center .hero-dark }

A fun **"anti-framework"** for doing useful things with agents and LLMs.
{ .hero-tagline }

<p align="center">
<a href="https://pypi.org/project/fastapi" target="_blank">
    <img src="https://img.shields.io/pypi/v/zyx?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
</p>

---

!!! note "What is `ZYX`?"

    `ZYX` is a simplicity-first library wrapped on top of [Pydantic AI]{ data-preview } and heavily inspired by [Marvin]{ data-preview }. It aims to provide a simple, *stdlib-like* interface for working with language models, without loss of control as well as flexibility that more complex frameworks provide.

## Introduction

`ZYX` stands on the shoulders of [Pydantic AI]{ data-preview } to provide a set of functions and components that aim to complete the following goals:

- **Simplicity-First**: Above all else, `ZYX` aims to be as simple as possible to use. The library is designed in a very semantically literal sense, with the hope that a 12 year old could pick up and run with it immediately.
- **Fast & Flexible**: `ZYX` is designed to be as fast as possible in two senses, (1) performance wise, and (2) development time wise, providing a very flexible interface that allows rapid prototyping and iteration.
- **Type-Focused**: `ZYX` provides a very type-focused interface, leveraging [Pydantic]'s powerful validation capabilities to ensure that your data is always in the expected format.
- **Model Agnostic**: Through [Pydantic AI]{ data-preview }, `ZYX` is completely model agnostic and supports virtually *any* LLM provider.

[Pydantic AI]: https://ai.pydantic.dev/ "A Python framework for building production-grade applications with LLMs"
[Marvin]: https://askmarvin.ai/ "A clean & simple frameworks for building AI applications"
