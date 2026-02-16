"""tests.test_graph_nodes"""

import asyncio
import pytest

from pydantic_graph.beta.step import StepContext
from pydantic_graph.nodes import BaseNode, End

from zyx._graph._nodes import run_v1_node_chain


class NodeB(BaseNode[object, object, str]):
    async def run(self, ctx):
        return End("done")


class NodeA(BaseNode[object, object, str]):
    async def run(self, ctx):
        return NodeB()


class BadNode(BaseNode[object, object, str]):
    async def run(self, ctx):
        return "not-a-node"


def test_run_v1_node_chain():
    ctx = StepContext(state=object(), deps=object(), inputs=None)
    result = asyncio.run(run_v1_node_chain(NodeA(), ctx)) # type: ignore[arg-type]
    assert result == "done"


def test_run_v1_node_chain_invalid_transition():
    ctx = StepContext(state=object(), deps=object(), inputs=None)
    with pytest.raises(ValueError):
        asyncio.run(run_v1_node_chain(BadNode(), ctx)) # type: ignore[arg-type]
