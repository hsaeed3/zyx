"""tests.test_targets"""

from zyx.targets import Target, target


def test_target_on_field_and_on():
    t = Target(target=str)

    @t.on_field("name")
    def _on_name(value):
        return value

    @t.on_field()
    def _on_self(value):
        return value

    @t.on("complete")
    def _on_complete(value):
        return value

    assert "name" in t._field_hooks
    assert "__self__" in t._field_hooks
    assert "complete" in t._prebuilt_hooks


def test_target_factory():
    t = target(int, name="Age", description="age")
    assert t.target is int
    assert t.name == "Age"
    assert t.description == "age"
