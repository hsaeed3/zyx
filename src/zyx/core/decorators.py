# zyx ==============================================================================

__all__ = [
    "batch",
    "lightning"
]

from typing import Callable, Iterable, Optional, Any, TypeVar, List, Generator
import time

T = TypeVar("T")
R = TypeVar("R")

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
