# zyx ==============================================================================

import concurrent.futures
import functools
import psutil
import time
from typing import Callable, Iterable, Optional, Any, TypeVar, List, Generator

T = TypeVar("T")
R = TypeVar("R")


def lightning(
    batch_size: Optional[int] = None,
    verbose: bool = False,
    timeout: Optional[float] = None,
) -> Callable[[Callable[[T], R]], Callable[[Iterable[T]], List[R]]]:
    """
    A decorator for high-performance parallel processing of CPU-bound tasks.

    This decorator can be used to automatically parallelize the execution of a function
    over an iterable input (including generators), with optional batching for improved performance.

    Args:
        batch_size (Optional[int]): The size of each batch. If None, no batching is performed.
        verbose (bool): Whether to print execution time and memory usage information.
        timeout (Optional[float]): The maximum number of seconds to wait for all tasks to complete.

    Returns:
        Callable: A decorator function that wraps the original function with parallel processing capabilities.

    Example:
        @zyx.lightning(batch_size=10, verbose=True)
        def square(x: int) -> int:
            return x * x

        results = square(range(100))
    """

    def decorator(func: Callable[[T], R]) -> Callable[[Iterable[T]], List[R]]:
        @functools.wraps(func)
        def wrapper(iterable: Iterable[T], *args: Any, **kwargs: Any) -> List[R]:
            num_workers = psutil.cpu_count(logical=False) or 4

            def process_batch(batch: List[T]) -> List[R]:
                return [func(item, *args, **kwargs) for item in batch]

            results: List[R] = []
            start_time = time.time() if verbose else None
            start_memory = psutil.virtual_memory().used if verbose else None

            # Convert generator to list if batching is required
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
