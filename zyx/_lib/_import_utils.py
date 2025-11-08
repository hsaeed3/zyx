"""zyx._lib._import_utils"""

from __future__ import annotations

import ast
import hashlib
import inspect
from importlib import import_module
from typing import Any, Callable, Dict, FrozenSet, List, OrderedDict, Tuple

from ._logging import _get_logger

# `_import_utils` is always the first module to be initialized
# within `zyx`, so logging will always be setup here
_get_logger()


__all__ = (
    "type_checking_getattr_fn",
    "type_checking_dir_fn",
)


class GetAttrFunctionError(AttributeError):
    """
    Error raised by the `type_checking_getattr_fn` function when it
    fails to resolve or import a requested attribute. Inherits from
    AttributeError for better integration with Python's import system.
    """


class GetAttrFunctionWarning(Warning):
    """
    Warning raised by the `type_checking_getattr_fn` function for
    non-critical issues.
    """


class GetAttrFunctionCache:
    """Minimal in-memory LRU cache implementation."""

    __slots__ = ("cache", "max_size")

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def make_cache_key(self, key: Any) -> str:
        """Creates a stable hash key from various inputs."""
        return hashlib.sha256(repr(key).encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        value = self.cache.get(key, default)
        if value is not default:
            self.cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def cached(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to cache function calls based on their arguments."""
        sentinel = object()

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key_repr = (args, tuple(sorted(kwargs.items())))
            key = self.make_cache_key(key_repr)
            result = self.get(key, sentinel)
            if result is sentinel:
                result = func(*args, **kwargs)
                self.set(key, result)
            return result

        return wrapper


_parser_cache = GetAttrFunctionCache(max_size=64)
_loader_cache: Dict[str, Tuple[Callable[[str], Any], Callable[[], List[str]]]] = {}
_CACHE_VERSION = "3.1.0"


@_parser_cache.cached
def _parse_type_checking_imports(source: str) -> Dict[str, Tuple[str, str]]:
    """
    Parses 'TYPE_CHECKING' blocks in source code to map local names
    to their import source (module_path, original_name).
    """
    tree = ast.parse(source)
    imports: Dict[str, Tuple[str, str]] = {}

    for node in ast.walk(tree):
        is_type_checking_block = False
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                is_type_checking_block = True
            elif (
                isinstance(test, ast.Attribute)
                and isinstance(test.value, ast.Name)
                and test.value.id == "typing"
                and test.attr == "TYPE_CHECKING"
            ):
                is_type_checking_block = True

        if not is_type_checking_block:
            continue

        for statement in node.body:
            if isinstance(statement, ast.ImportFrom) and statement.module:
                module_path = (
                    "." * statement.level + statement.module
                    if statement.level > 0
                    else statement.module
                )

                for name in statement.names:
                    original_name = name.name
                    local_name = name.asname or original_name
                    imports[local_name] = (module_path, original_name)

    return imports


def _create_lazy_loaders(
    import_map: Dict[str, Tuple[str, str]],
    package: str | None,
    all_attributes: FrozenSet[str],
) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Generates and returns the `__getattr__` and `__dir__` functions."""
    _import_cache: Dict[str, Any] = {}

    def __getattr__(name: str) -> Any:
        if name in _import_cache:
            return _import_cache[name]

        import_info = import_map.get(name)
        if import_info:
            module_path, original_name = import_info
            try:
                # Differentiate between relative and absolute imports
                if module_path.startswith("."):
                    # Relative imports need the 'package' argument
                    module = import_module(module_path, package)
                else:
                    # Absolute imports do not
                    module = import_module(module_path)

                attribute = getattr(module, original_name)
                _import_cache[name] = attribute
                return attribute
            except (ImportError, AttributeError) as e:
                raise GetAttrFunctionError(
                    f"Failed to lazy-load '{name}' from '{module_path}'. Reason: {e}"
                ) from e

        if name in all_attributes:
            try:
                module = import_module(f".{name}", package)
                _import_cache[name] = module
                return module
            except ImportError:
                pass

        raise GetAttrFunctionError(
            f"Module '{package or '__main__'}' has no attribute '{name}'"
        )

    def __dir__() -> List[str]:
        """Provides support for `dir()` and autocompletion."""
        return sorted(list(all_attributes))

    return __getattr__, __dir__


def _get_loader_functions(
    all_list: Tuple[str, ...] | List[str],
) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Factory to inspect the calling module and return loader functions."""
    try:
        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Cannot determine calling module's frame.")

        # Walk up the stack to find the first frame outside this module
        current_module = __name__
        caller_frame = frame.f_back

        while caller_frame is not None:
            frame_globals = caller_frame.f_globals
            frame_module = frame_globals.get("__name__")

            # Found a frame from a different module - this is our caller
            if frame_module != current_module:
                caller_globals = frame_globals
                break

            caller_frame = caller_frame.f_back
        else:
            raise RuntimeError("Cannot determine calling module's frame.")
    finally:
        del frame

    package = caller_globals.get("__package__")
    filename = caller_globals.get("__file__")

    if not filename:
        raise RuntimeError("Cannot find source file for the calling module.")

    cache_key = _CACHE_VERSION + filename
    if cache_key in _loader_cache:
        return _loader_cache[cache_key]

    try:
        with open(filename, "r", encoding="utf-8") as f:
            source = f.read()
    except (IOError, OSError) as e:
        raise RuntimeError(f"Cannot read source file: {filename}") from e

    import_map = _parse_type_checking_imports(source)
    all_set = frozenset(all_list)

    filtered_map = {name: info for name, info in import_map.items() if name in all_set}

    loader_tuple = _create_lazy_loaders(filtered_map, package, all_set)
    _loader_cache[cache_key] = loader_tuple
    return loader_tuple


def type_checking_getattr_fn(
    all_list: Tuple[str, ...] | List[str],
) -> Callable[[str], Any]:
    """
    Auto-generates a `__getattr__` function for a module to enable lazy loading.

    It parses `if TYPE_CHECKING:` blocks in the calling module's source to
    discover how to import attributes listed in `__all__`.

    Example:

        ### my_package/__init__.py

        ```python
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            # any relative import
            from .submodule import MyClass
            # or external import
            from os import PathLike

        __all__ = ("MyClass", "PathLike")

        __getattr__ = type_checking_getattr_fn(__all__)
        ```

    Args:
        all_list: The `__all__` list from the calling module.

    Returns:
        A PEP 562 compliant `__getattr__` function.
    """

    getattr_func, _ = _get_loader_functions(all_list)
    return getattr_func


def type_checking_dir_fn(
    all_list: Tuple[str, ...] | List[str],
) -> Callable[[], List[str]]:
    """
    Auto-generates a `__dir__` function to accompany `type_checking_getattr_fn`.

    A module that defines `__getattr__` should also define `__dir__` to
    ensure proper support for introspection and autocompletion.

    Args:
        all_list: The `__all__` list from the calling module, which should be
                  identical to the one passed to `type_checking_getattr_fn`.

    Returns:
        A PEP 562 compliant `__dir__` function.
    """
    _, dir_func = _get_loader_functions(all_list)
    return dir_func
