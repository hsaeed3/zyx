# AGENTS.md — zyx core

This file orients coding agents to the public surface area and conventions in `zyx`. Use it when reading or modifying the library. Do not rely on the docs site yet; the source code is the reference of truth.

## Quick mental model

`zyx` is a thin semantic-operations layer on top of `pydantic_ai`. The core 'functionality' is through something called a **semantic operation**.

- A `source` (primary input) (depends on the kind of operation, things like `make` dont include this parameter)
- A `target` (type or value you want)
- Optional `context`, `instructions`, `tools`, `attachments`, and `deps`

The library builds a semantic graph and returns a `Result[T]` or `Stream[T]` that wraps a typed output plus raw run metadata.

## Core objects

### Fundamental Objects & Concepts

The entire generative interface of `zyx` is built around `zyx._graph` which uses `pydantic_graph` to subclass/configure a series of nodes within `zyx._graph._nodes` that make up the execution graph of a semantic operation, following by operation specific usage of the objects in `zyx._graph._context` and `zyx._graph._requests`.

Finally at the operation level, use `zyx._graph` to create a `SemanticGraph` object that can be executed to produce a `Result[T]` or `Stream[T]`.

Key graph primitives:

- `SemanticGraphDeps`: immutable, normalized deps/config for a run.
- `SemanticGraphState`: mutable run state, output builder, and stream tracking.
- `SemanticGraphContext`: per-node view of deps + state.
- `SemanticGraphRequestTemplate`: builds a single model request from deps/state.

`SemanticGraph` executes nodes and always returns a `Result[T]` or `Stream[T]`. It also auto-updates any `Context` passed in `context` if `Context.update=True`.

### Result

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/result.py`

`Result[T]` is the universal wrapper for a completed semantic operation.

- `output`: the final typed value.
- `raw`: list of underlying `pydantic_ai` agent run results.
- `confidence`: computed from log-probs if enabled/supported.
- `model`: model name from the first run.
- `usage`: aggregated token usage across all runs.
- `all_new_messages`: all messages generated during the operation.

### Stream

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/stream.py`

`Stream[T]` is the streaming wrapper for semantic operations.

- Provides async streaming methods: `stream_text`, `stream_partial`, `stream_field`.
- Provides sync wrappers: `text`, `partial`, `field`.
- Finalization: `finish()` / `finish_async()` returns `Result[T]`.
- `result` property is only available after the stream completes.
- Internally updates an `OutputBuilder` as streams complete; supports `exclude_none` for selective edits.

### Resources

Files:

- `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/resources/abstract.py`
- `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/resources/file.py`
- `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/resources/memory/__init__.py`

Resources are tool- and attachment-compatible objects that models can read, query, or mutate.

Core contract (`AbstractResource`):

- `get_description()`: human-readable resource description.
- `get_state_description()`: current state snapshot (e.g., file contents).
- `get_toolset()`: `pydantic_ai` toolset to interact with the resource.

Semantics:

- If a resource is in `tools`, only its toolset is available.
- If a resource is in `attachments`, the toolset plus a description/state snapshot is injected.

`File`:

- Represents a filesystem file (text/JSON/YAML/TOML/INI/etc).
- Uses anchor-based edits for mutation and can hard-replace or append.
- `writeable` and `confirm` guard mutations; `confirm=True` blocks writes.
- Truncates read content beyond `max_chars` and uses a per-file lock.

`Memory`:

- Vector-store-backed resource for add/search/delete.
- Providers: `chroma/persistent`, `chroma/ephemeral`, `qdrant`.
- `auto` controls whether it should auto-add memories when used by operations.
- Exposes toolset: `add_memory`, `delete_memory`, `search_memory`.

### Context

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/context.py`

`Context` is a mutable container for conversation history, instructions, tools, and deps. It is the only object that automatically updates across operations.

Key behaviors:

- `Context.update` controls whether operations write their message history back into the context.
- `compact_instructions` merges all instructions into a single system prompt part.
- `exclude_messages`, `exclude_instructions`, `exclude_tools` prevent forwarding those elements.
- `max_length` truncates forwarded history from the tail.

Construction patterns:

- `Context(...)` for explicit construction.
- `create_context(...)` as a functional helper.
- `ctx(...)` returns a copied context with per-call overrides.

### Snippet / paste

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/snippets.py`

`Snippet` wraps a source into a multimodal-aware object that can be inserted into context or used as an attachment.

Supported source types:

- Path-like files
- URLs
- Bytes
- Arbitrary objects that can be encoded to TOON text

Use `paste(source)` as the convenience constructor.

### Target / target

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/targets.py`

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

## Semantic operations

The main entry points are in `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/`.

All operations accept the same core parameter shape and return `Result[T]` (or `Stream[T]` for streaming).

- `make` / `amake`: Generate new content for a target type or value.
- `parse` / `aparse`: Extract structured content from a primary source.
- `query` / `aquery`: Answer questions grounded in the primary source.
- `edit` / `aedit`: Modify a target value or object, optionally selective or planned.
- `select` / `aselect`: Choose from a list/enum/literal set.
- `validate` / `avalidate`: Parse then verify constraints; can return structured violations.
- `expr`: Semantic expression helpers (`==`, `in`, `bool`) using `parse`.

## Common parameter types

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/_types.py`

These are the canonical inputs for operation signatures.

- `SourceParam`: any value used as primary input.
- `TargetParam[T]`: `T`, `type[T]`, or `Target[T]`.
- `ContextType`: `str`, `pydantic_ai` message, Pydantic model, `dict`, `Context`, or `Snippet`.
- `ModelParam`: `pydantic_ai` agent, model, known model name string, or provider string.
- `ToolType`: function, `pydantic_ai` tool, builtin tool, toolset, or `zyx` resource.
- `AttachmentType`: `Snippet` or `AbstractResource`.

## Parameter conventions

Shared across operations:

- `context`: optional conversation history or `Context`.
- `instructions`: extra system guidance.
- `tools`: available tools for tool-using models/agents.
- `attachments`: persistent content placed to avoid context rot and to enable resource-style tool interactions.
- `deps`: passthrough to `pydantic_ai.RunContext` dependencies.
- `model_settings`: forwarded to the underlying model (e.g., temperature).
- `usage_limits`: token/request limits.
- `confidence`: enables log-probability confidence scores where supported.
- `stream`: use streaming and return a `Stream[T]`.

Request composition details (from `_graph`):

- `SemanticGraphDeps.prepare(...)` normalizes `context`, `instructions`, `tools`, `attachments`, and builds a `pydantic_ai.Agent`.
- `Context` objects in `context` contribute messages, toolsets, and rendered instructions.
- If `confidence=True`, deps ensure a confidence-capable agent.
- `SemanticGraphRequestTemplate.render(...)` merges:
  - Base message history
  - Prior run messages (if enabled)
  - Target instructions (if the target is a `Target`)
  - Dynamic `system_prompt_additions`
  - Attachments and optional source context
  - Output context (for edit-style operations)
  - Optional user prompt additions
  - Toolsets (user + per-request)

Source and attachments:

- `source` is treated as ground truth for `parse` and `query` and is rendered into system prompt context.
- If `source` is a `Snippet` or a `File` resource, its content may be included as a multimodal attachment.
- `attachments` are persistent context with optional resource toolsets; snippets are inserted into message history.

## Operation-specific notes

### make

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/make.py`

Requires `context` or `instructions` when `target=str`, otherwise it errors. It can optionally inject a randomization prompt for diverse outputs.

### parse

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/parse.py`

Strict “extract only” behavior. The parse system prompt forbids following instructions inside the primary input.

### query

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/query.py`

Grounded response. The system prompt mandates using only primary input; schema-only output.

### edit

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/edit/__init__.py`

Supports selective edits, edit plans, iterative edits, and merge behavior for mapping-like targets. If `merge=True`, target must be mapping-like.

### select

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/select.py`

Accepts Literals, Enums, lists, unions. Internally uses a structured selection model and maps back to the original choice.

### validate

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/validate.py`

Performs parse, then constraint validation. `raise_on_error=False` returns `ValidationResult[T]` with violations.

### expr

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/operations/expressions.py`

Semantic expressions are backed by `parse`, not pure heuristics. Treat them as model calls.

## Streaming

Streaming operations return `Stream[T]` and expose:

- `text(delta=False, debounce_by=0.1)`
- `partial(debounce_by=0.1)`
- `field(field_name, debounce_by=0.1)`
- `is_streaming`, `usage`

Some operations wrap streams to remap structured results back to target objects.

## Guidance for changes

- Prefer editing the semantic operation implementation rather than re-stating behavior in docs for now.
- Keep public API signatures consistent across `make`, `parse`, `query`, `edit`, `select`, `validate`.
- Use `_types.py` aliases in public interfaces.
- Avoid adding new top-level concepts unless they are provider-agnostic and reused across at least two operations.

## Entry points

File: `/Users/hammad/Development/ZYX_DEV/zyx-core/zyx/__init__.py`

Top-level exports:

- Core: `Context`, `create_context`, `Snippet`, `paste`, `Target`, `target`
- Resources: `Code`, `File`, `Memory`
- Ops: `make`, `amake`, `edit`, `aedit`, `expr`, `parse`, `aparse`, `query`, `aquery`, `select`, `aselect`, `validate`, `avalidate`
