# zyx ==============================================================================

__all__ = [
    "BaseModel",
    "Field",
    "logger",
    "rich_console"
    "_UtilLazyLoader"
    "batch",
    "lightning",
]

from pydantic.main import BaseModel
from pydantic.fields import Field
from importlib import import_module
from types import ModuleType
import threading
import inspect
from typing import Callable, Iterable, Optional, Any, TypeVar, List, Generator
import time

T = TypeVar("T")
R = TypeVar("R")

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

class _UtilLazyLoader(metaclass=LazyLoaderMeta):
    """
    Lazy loader for a module.
    """
    _module_cache: Optional[ModuleType] = None
    _lock = threading.Lock()

    @classmethod
    def _load(cls) -> Any:
        if cls._module_cache is None:
            with cls._lock:
                if cls._module_cache is None:
                    try:
                        module = import_module(cls._module_name)
                        cls._module_cache = getattr(module, cls._attr_name)
                        
                        # Handle special cases
                        if inspect.isclass(cls._module_cache):
                            return type(cls.__name__, (cls._module_cache,), {})
                        elif inspect.isgeneratorfunction(cls._module_cache):
                            return cls._wrap_generator(cls._module_cache)
                        elif inspect.isfunction(cls._module_cache):
                            return cls._wrap_function(cls._module_cache)
                    except (ImportError, AttributeError) as e:
                        raise e
        return cls._module_cache

    @staticmethod
    def _wrap_generator(gen_func):
        def wrapper(*args, **kwargs):
            return gen_func(*args, **kwargs)
        return wrapper

    @staticmethod
    def _wrap_function(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @classmethod
    def init(cls, module_name: str, attr_name: str) -> None:
        cls._module_name = module_name
        cls._attr_name = attr_name

    def __getattr__(self, name: str) -> Any:
        module = self._load()
        try:
            return getattr(module, name)
        except AttributeError as e:
            raise e

    def __iter__(self):
        module = self._load()
        if hasattr(module, '__iter__'):
            return iter(module)
        raise TypeError(f"{module} object is not iterable")

    def __next__(self):
        module = self._load()
        if hasattr(module, '__next__'):
            return next(module)
        raise TypeError(f"{module} object is not an iterator")

    def __getitem__(self, item):
        module = self._load()
        if hasattr(module, '__getitem__'):
            return module[item]
        raise TypeError(f"{module} object does not support indexing")

    def __len__(self):
        module = self._load()
        if hasattr(module, '__len__'):
            return len(module)
        raise TypeError(f"{module} object does not have a length")

    def __contains__(self, item):
        module = self._load()
        if hasattr(module, '__contains__'):
            return item in module
        raise TypeError(f"{module} object does not support membership test")
    
# ==============================================================================

# zyx ==============================================================================

def batch(
    batch_size: Optional[int] = None,
    verbose: bool = False,
    timeout: Optional[float] = None,
) -> Callable[[Callable[[T], R]], Callable[[Iterable[T]], List[R]]]:
    import asyncio
    import functools
    def decorator(func: Callable[[T], R]) -> Callable[[Iterable[T]], List[R]]:
        @functools.wraps(func)
        async def wrapper(iterable: Iterable[T], *args: Any, **kwargs: Any) -> List[R]:
            num_workers = psutil.cpu_count(logical=False) or 4

            async def process_batch(batch: List[T]) -> List[R]:
                return await asyncio.gather(*[func(item, *args, **kwargs) for item in batch])

            results: List[R] = []
            start_time = time.time() if verbose else None
            start_memory = psutil.virtual_memory().used if verbose else None

            if batch_size and isinstance(iterable, Generator):
                iterable = list(iterable)

            if batch_size:
                batches = [
                    list(iterable)[i : i + batch_size]
                    for i in range(0, len(list(iterable)), batch_size)
                ]
            else:
                batches = [[item] for item in iterable]

            semaphore = asyncio.Semaphore(num_workers)

            async def process_with_semaphore(batch):
                async with semaphore:
                    return await process_batch(batch)

            tasks = [process_with_semaphore(batch) for batch in batches]

            try:
                for task in asyncio.as_completed(tasks, timeout=timeout):
                    batch_results = await task
                    results.extend(batch_results)
            except asyncio.TimeoutError:
                print("Operation timed out")

            if verbose:
                import psutil
                end_time = time.time()
                end_memory = psutil.virtual_memory().used
                execution_time = end_time - start_time if start_time else 0
                memory_used = end_memory - start_memory if start_memory else 0
                print(f"Execution Time: {execution_time:.2f} seconds")
                print(f"Memory Used: {memory_used} bytes")
                print(f"Worker Threads Used: {num_workers}")
                print(f"Available Memory: {psutil.virtual_memory().available} bytes")
                print(f"Total Memory: {psutil.virtual_memory().total} bytes")

            return results

        return wrapper

    return decorator

def lightning(
    batch_size: Optional[int] = None,
    verbose: bool = False,
    timeout: Optional[float] = None,
) -> Callable[[Callable[[T], R]], Callable[[Iterable[T]], List[R]]]:
    import functools
    def decorator(func: Callable[[T], R]) -> Callable[[Iterable[T]], List[R]]:
        @functools.wraps(func)
        def wrapper(iterable: Iterable[T], *args: Any, **kwargs: Any) -> List[R]:
            import concurrent.futures
            import psutil
            num_workers = psutil.cpu_count(logical=False) or 4

            def process_batch(batch: List[T]) -> List[R]:
                return [func(item, *args, **kwargs) for item in batch]

            results: List[R] = []
            start_time = time.time() if verbose else None
            start_memory = psutil.virtual_memory().used if verbose else None

            if batch_size and isinstance(iterable, Generator):
                iterable = list(iterable)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                if batch_size:
                    batches = [
                        list(iterable)[i : i + batch_size]
                        for i in range(0, len(list(iterable)), batch_size)
                    ]
                    future_to_batch = {
                        executor.submit(process_batch, batch): batch
                        for batch in batches
                    }
                else:
                    future_to_item = {
                        executor.submit(func, item, *args, **kwargs): item
                        for item in iterable
                    }

                for future in concurrent.futures.as_completed(
                    future_to_batch if batch_size else future_to_item, timeout=timeout
                ):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results if batch_size else [batch_results])
                    except Exception as exc:
                        item = (
                            future_to_batch[future]
                            if batch_size
                            else future_to_item[future]
                        )
                        print(f"Generated an exception: {exc} with item/batch: {item}")

            if verbose:
                end_time = time.time()
                end_memory = psutil.virtual_memory().used
                execution_time = end_time - start_time if start_time else 0
                memory_used = end_memory - start_memory if start_memory else 0
                print(f"Execution Time: {execution_time:.2f} seconds")
                print(f"Memory Used: {memory_used} bytes")
                print(f"Worker Threads Used: {num_workers}")
                print(f"Available Memory: {psutil.virtual_memory().available} bytes")
                print(f"Total Memory: {psutil.virtual_memory().total} bytes")

            return results

        return wrapper

    return decorator

# ==============================================================================

class _logger(_UtilLazyLoader):
    pass
_logger.init("loguru", "logger")

class _rich_console(_UtilLazyLoader):
    pass
_rich_console.init("rich.console", "Console")

class _batch(_UtilLazyLoader):
    pass
_batch.init("zyx.core", "batch")

class _lightning(_UtilLazyLoader):
    pass
_lightning.init("zyx.core", "lightning")
