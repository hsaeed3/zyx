# zyx ===============================================

from importlib import import_module
from types import ModuleType
from typing import Any, Optional
import threading


class LazyLoaderMeta(type):
    """
    Meta class for lazy loading attributes from a module.
    """

    def __getattr__(cls, name: str) -> Any:
        module = cls._load()
        try:
            return getattr(module, name)
        except AttributeError as e:
            raise e

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        module = cls._load()
        return module(*args, **kwargs)


class UtilLazyLoader(metaclass=LazyLoaderMeta):
    """
    Lazy loader for a module.
    """

    _module_cache: Optional[ModuleType] = None
    _lock = threading.Lock()

    @classmethod
    def _load(cls) -> ModuleType:
        if cls._module_cache is None:
            with cls._lock:
                if cls._module_cache is None:
                    try:
                        module = import_module(cls._module_name)
                        cls._module_cache = getattr(module, cls._attr_name)
                    except (ImportError, AttributeError) as e:
                        raise e
        return cls._module_cache

    @classmethod
    def init(cls, module_name: str, attr_name: str) -> None:
        cls._module_name = module_name
        cls._attr_name = attr_name


# ==============================================================================
