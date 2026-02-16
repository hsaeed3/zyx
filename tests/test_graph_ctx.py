"""tests.test_graph_ctx"""

from zyx._graph._ctx import InjectedDeps


class UserDeps:
    def __init__(self):
        self.foo = "bar"


def test_injected_deps_getattr_and_get():
    user = UserDeps()
    deps = InjectedDeps(user=user, internal={"baz": 1})
    assert deps.foo == "bar"
    assert deps.baz == 1
    assert deps.get("missing", "default") == "default"
