import concurrent.futures
from typing import Callable, List, Any, Tuple
from tqdm import tqdm


def process_in_parallel(
    items: List[Tuple[Any, ...]],
    process_func: Callable,
    max_workers: int = 10,
    show_progress: bool = True
) -> List[dict]:
    """
    Process items in parallel using ThreadPoolExecutor with tqdm progress.

    Args:
        items: List of tuples, each containing arguments for process_func
        process_func: Function to process each item
        max_workers: Maximum number of parallel threads
        show_progress: Whether to show tqdm progress bar

    Returns:
        List of results ordered by the first element of each item tuple (assumed to be index)
    """
    total = len(items)
    results_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a mapping from future to index
        future_to_index = {}
        for item in items:
            # First element is assumed to be the index
            index = item[0]
            future = executor.submit(process_func, *item)
            future_to_index[future] = index

        # Collect results as they complete
        progress = tqdm(total=total, desc="Processing", unit="sent", disable=not show_progress)
        try:
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                results_dict[index] = result
                progress.update(1)
        finally:
            progress.close()

    # Sort results by index to maintain order
    results = [results_dict[i] for i in sorted(results_dict.keys())]

    return results
