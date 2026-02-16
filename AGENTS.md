# AGENTS.md — zyx core

This file orients coding agents to the public surface area and conventions in `zyx`. Use it when reading or modifying the library. The source code is the reference of truth.

## Quick mental model

`zyx` is a thin semantic-operations layer on top of `pydantic_ai`. The core functionality is a **semantic operation**.

Each operation accepts:

- `source` (primary input, when applicable)
- `target` (type or value you want)
- Optional `context`, `instructions`, `tools`, `attachments`, and `deps`

The library builds a semantic graph and returns a `Result[T]` or `Stream[T]` wrapper.

## Core objects

### Semantic graph

`zyx._graph` is the execution engine. It uses `pydantic_graph.beta` steps to build and run a graph for each operation.

Key graph primitives:

- `SemanticGraphDeps`: immutable, normalized deps/config for a run
- `SemanticGraphState`: mutable run state, output builder, and stream tracking
- `SemanticGraphContext`: per-step view of deps + state
- `SemanticGraphRequestTemplate`: builds a single model request from deps/state

`SemanticGraph` executes steps and returns `Result[T]` or `Stream[T]`. It also auto-updates any `Context` passed in `context` if `Context.update=True`.

### Result

File: `zyx/result.py`

`Result[T]` is the universal wrapper for a completed semantic operation.

- `output`: final typed value
- `raw`: list of underlying `pydantic_ai` agent run results
- `confidence`: computed from log-probs if enabled/supported
- `model`: model name from the first run
- `usage`: aggregated token usage across all runs
- `all_new_messages`: all messages generated during the operation
- `as_semantic_message()`: semantic summary for context updates

### Stream

File: `zyx/stream.py`

`Stream[T]` is the streaming wrapper for semantic operations.

- Async streaming: `stream_text`, `stream_partial`, `stream_field`
- Sync wrappers: `text`, `partial`, `field`
- Finalization: `finish()` / `finish_async()` returns `Result[T]`
- `result` property is available after stream completion

Internally updates an `OutputBuilder` as streams complete; supports `exclude_none` for selective edits.

### Context

File: `zyx/context.py`

`Context` is a mutable container for conversation history, instructions, tools, and deps. It is the only object that automatically updates across operations.

Key behaviors:

- `Context.update` controls whether operations write their semantic messages back into the context
- User-provided message history in `context=[...]` is appended first, then the semantic message
- `compact_instructions` merges all instructions into a single system prompt part
- `exclude_messages`, `exclude_instructions`, `exclude_tools` prevent forwarding those elements
- `max_length` truncates forwarded history from the tail

Construction patterns:

- `Context(...)` for explicit construction
- `create_context(...)` as a functional helper
- `ctx(...)` returns a copied context with per-call overrides

### Attachments

Files:

- `zyx/attachments/__init__.py`
- `zyx/_strategies/_attachments.py`

Attachments provide persistent context and (optionally) tool interfaces. They are used to avoid context rot and to enable structured interaction with files or objects.

- `Attachment(source, interactive=False, writeable=False, ...)`
- `attach(source, writeable=True, ...)` for interactive attachments
- `paste_attachment(source, ...)` for static attachments

File attachments support anchor-based edits via `AnchorEdit` from `zyx._strategies._attachments`.

### Target

File: `zyx/targets.py`

`Target[T]` describes the output schema and guidance for semantic operations.

Core fields:

- `target`: type or value
- `name`, `description`: human-friendly hints
- `instructions`: per-target instructions (merged into call instructions)
- `constraints`: used by `validate` and `Target.validate`
- `model`: default model/agent for this target

Hooks:

- `@target.on_field(field=..., retry=True, update=False)` for field-level hooks
- `@target.on(event="complete" | "error", retry=True, update=False)` for lifecycle hooks

Use `target(...)` to create a `Target` from a type or value.

## Strategies

Files:

- `zyx/_strategies/_abstract.py`
- `zyx/_strategies/_objects.py`
- `zyx/_strategies/_edit.py`

Strategies unify how objects are described, represented, and mutated.

- `SourceStrategy` handles `source` rendering and metadata for requests.
- `TargetStrategy` captures target metadata and hook flags for request output selection.
- `EditStrategy` selects edit behavior based on the output builder.

## Semantic operations

Entry points live in `zyx/operations/`.

All operations accept the same core parameter shape and return `Result[T]` (or `Stream[T]` for streaming).

- `make` / `amake`: Generate new content for a target type or value.
- `parse` / `aparse`: Extract structured content from a primary source.
- `query` / `aquery`: Answer questions grounded in the primary source.
- `edit` / `aedit`: Modify a target value or object, optionally selective or planned.
- `select` / `aselect`: Choose from a list/enum/literal set.
- `validate` / `avalidate`: Parse then verify constraints; can return structured violations.
- `expr`: Semantic expression helpers (`==`, `in`, `bool`) using `parse`.
- `run` / `arun`: Tool-verified task execution with optional planning.

## Common parameter types

File: `zyx/_types.py`

- `SourceParam`: any value used as primary input
- `TargetParam[T]`: `T`, `type[T]`, or `Target[T]`
- `ContextType`: `str`, `pydantic_ai` message, Pydantic model, `dict`, or `Context`
- `ModelParam`: `pydantic_ai` agent, model, known model name string, or provider string
- `ToolType`: function, `pydantic_ai` tool, builtin tool, or toolset (or any object exposing `get_toolset()`)
- `AttachmentType`: `Attachment`

## Parameter conventions

Shared across operations:

- `context`: optional conversation history or `Context`
- `instructions`: extra system guidance
- `tools`: available tools for tool-using models/agents
- `attachments`: persistent content placed to avoid context rot
- `deps`: passthrough to `pydantic_ai.RunContext` dependencies
- `model_settings`: forwarded to the underlying model (e.g., temperature)
- `usage_limits`: token/request limits
- `confidence`: enables log-probability confidence scores where supported
- `stream`: use streaming and return a `Stream[T]`

Request composition details (from `_graph`):

- `SemanticGraphDeps.prepare(...)` normalizes `context`, `instructions`, `tools`, `attachments`, and builds a `pydantic_ai.Agent`
- `Context` objects in `context` contribute messages, toolsets, and rendered instructions
- `SemanticGraphRequestTemplate.render(...)` merges:
  - Base message history
  - Prior run messages (if enabled)
  - Target instructions (if any)
  - Dynamic `system_prompt_additions`
  - Attachments and optional source context
  - Output context (for edit-style operations)
  - Optional user prompt additions
  - Toolsets (user + per-request)

## Streaming

Streaming operations return `Stream[T]` and expose:

- `text(delta=False, debounce_by=0.1)`
- `partial(debounce_by=0.1)`
- `field(field_name, debounce_by=0.1)`

## Entry points

File: `zyx/__init__.py`

Top-level exports:

- Core: `Context`, `create_context`, `Attachment`, `attach`, `paste_attachment`, `Target`, `target`
- Tools: `Memory`
- Ops: `make`, `amake`, `edit`, `aedit`, `expr`, `parse`, `aparse`, `query`, `aquery`, `select`, `aselect`, `validate`, `avalidate`, `run`, `arun`

## Guidance for changes

- Prefer editing semantic operation implementations rather than restating behavior in docs.
- Keep public API signatures consistent across `make`, `parse`, `query`, `edit`, `select`, `validate`.
- Use `_types.py` aliases in public interfaces.
- Avoid adding new top-level concepts unless they are provider-agnostic and reused across at least two operations.
