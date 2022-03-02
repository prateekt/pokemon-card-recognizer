import multiprocessing
from typing import Callable, List, Any


def _sequential(func: Callable, params: List[Any]) -> List[Any]:
    """
    Sequentially runs a function.

    param func: The function to run
    param params: The parameters to the function
    return:
        Results of function executions
    """
    return [func(param) for param in params]


def _pool(func: Callable, params: List[Any]) -> List[Any]:
    """
    Implements paraloop using multiprocessing pool.

    param func: The function to parallelize
    param params: Inputs to function
    return:
        Results of function executions
    """
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cores) as p:
        return p.map(func, params)


def loop(func: Callable, params: List[Any], mechanism: str = "pool") -> List[Any]:
    """
    Executes function calls with list of parameters using specified mechanism.

    param func: The function
    param params: The parameters to use
    param mechanism: The mechanism
    return:
        Results of function executions
    """
    if mechanism == "pool":
        return _pool(func=func, params=params)
    elif mechanism == "sequential":
        return _sequential(func=func, params=params)
    else:
        raise ValueError("Unsupported mechanism: " + str(mechanism))
