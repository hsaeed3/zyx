"""
### zyx.core.tracing.events

Contains a 'type' helper Literal for all tracing event patterns available
in the application, as well as the pattern handler.

`zyx` follows this pattern when specifying specific events you want to trace:

```python
# This will display all traceable events for any agent used
trace('agent')

# This will display all traceable events for an agent named 'steve'
trace('agent(steve)')

# This will display all completion requests, for an agent named 'steve', ONLY
# if it is in the context of a graph.
tracing('graph:agent(steve):completion:request')
```
"""

from __future__ import annotations

from typing import Literal, Pattern, Dict, Set, get_args


# -----------------------------------------------------------------------------
# [Core Patterns]
# -----------------------------------------------------------------------------

TRACED_MODULE_PATTERN: Pattern = r"^(\w+)(?:\(([\w-]+)\))?$"
"""Regex pattern for modules ex: agent(steve)"""

TRACED_EVENT_PATTERN: Pattern = r"^(?:[\w-]+(?:\([\w-]+\))?:)*[\w-]+$"
"""Regex pattern for full event patterns ex: graph:agent(steve):completion:request"""


# -----------------------------------------------------------------------------
# [Modules & Events]
# NOTE: Currently zyx defines the module level events here as well. idk if this is the best
# way to do it, but it works rn.
# -----------------------------------------------------------------------------

# Core List of Traced Modules
TracedModuleType = Literal[
    # The graph API
    "graph",
    # The agent API
    "agent",
    # Chat Completions
    "completion",
    # State
    "state",
    # Actions (mostly used in the graph)
    "action",
    # memory (vectors)
    "memory",
]


# Different Event Types for Modules
# these do not pertain to 'graph:agent' for example, these are all direct
# to the module

# 'graph'
GraphTracedEventType = Literal[
    "create",
    "build",
    "run",
]

# 'agent'
AgentTracedEventType = Literal["create",]

# 'completions'
CompletionTracedEventType = Literal[
    "request",
    "response",
    # checks the last output or response if tool calls are provided
    "tool_call",
    "tool_output",
    "tool_error",
    # if the completion was retried
    "retry",
]

# 'state
StateTracedEventType = Literal[
    "create",
    "add",
    # for scratchpads
    "substate",
    "clear",
]

# `action`
# actions are all user defined
# ex action(run_code):execute
ActionTracedEventType = Literal["execute",]


# memory
MemoryTracedEventType = Literal["add", "delete", "update", "query"]


# union
TracedEventType = (
    GraphTracedEventType
    | AgentTracedEventType
    | CompletionTracedEventType
    | StateTracedEventType
    | ActionTracedEventType
    | MemoryTracedEventType
)


# -----------------------------------------------------------------------------
# [Registry]
# NOTE: this also has to be updated if the modules change... lol...
# -----------------------------------------------------------------------------

TRACED_MODULE_EVENTS: Dict[TracedModuleType, Set[TracedEventType]] = {
    "graph": set(GraphTracedEventType),
    "agent": set(AgentTracedEventType),
    "completion": set(CompletionTracedEventType),
    "state": set(StateTracedEventType),
    "action": set(ActionTracedEventType),
    "memory": set(MemoryTracedEventType),
}


# -----------------------------------------------------------------------------
# [Pattern Generation]
# -----------------------------------------------------------------------------

TRACED_MODULES = get_args(TracedModuleType)

# Generate Base Module Patterns
BASED_TRACED_MODULE_PATTERN = Literal[
    "graph",
    "agent",
    "completion",
    "state",
    "action",
    "memory",
]

# Entity Patterns
# all 'renamable' instances
TRACED_ENTITY_PATTERN = Literal["agent(*)", "tool(*)", "memory(*)", "state(*)", "action(*)", "graph(*)"]


# Event Patterns
# also need to be updated if the modules change... lol...
# this one is horrendous
TracedEventPattern = Literal[
    # -----------------------------------------------------------------------------
    # Core Level (Lowest) -- These have no sub-modules
    # -----------------------------------------------------------------------------
    "completion:request",
    "completion:response",
    "completion:tool_call",
    "completion:tool_output",
    "completion:tool_error",
    "completion:retry",
    "action:execute",
    "state:create",
    "state:add",
    "state:substate",
    "state:clear",
    "memory:add",
    "memory:delete",
    "memory:update",
    "memory:query",
    # -----------------------------------------------------------------------------
    # Agents
    # -----------------------------------------------------------------------------
    "agent:create",
    "agent:completion:request",
    "agent:completion:response",
    "agent:completion:tool_call",
    "agent:completion:tool_output",
    "agent:completion:tool_error",
    "agent:completion:retry",
    "agent:state:create",
    "agent:state:add",
    "agent:state:substate",
    "agent:state:clear",
    "agent:memory:add",
    "agent:memory:delete",
    "agent:memory:update",
    "agent:memory:query",
    # -----------------------------------------------------------------------------
    # Graphs
    # -----------------------------------------------------------------------------
    "graph:action:execute",
    "graph:state:create",
    "graph:state:add",
    "graph:state:substate",
    "graph:state:clear",
    "graph:memory:add",
    "graph:memory:delete",
    "graph:memory:update",
    "graph:memory:query",
    "graph:agent:create",
    "graph:agent:completion:request",
    "graph:agent:completion:response",
    "graph:agent:completion:tool_call",
    "graph:agent:completion:tool_output",
    "graph:agent:completion:tool_error",
    "graph:agent:completion:retry",
    "graph:agent:state:create",
    "graph:agent:state:add",
    "graph:agent:state:substate",
    "graph:agent:state:clear",
    "graph:agent:memory:add",
    "graph:agent:memory:delete",
    "graph:agent:memory:update",
    "graph:agent:memory:query",
]
