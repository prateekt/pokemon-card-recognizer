import copy
import queue
from functools import wraps
from typing import Dict, Any, List, Sequence, Callable

import numpy as np


def enumerate_settings(kwargs: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """
    Enumerates all possible parameter settings.

    param kwargs: Map of keyword args to sequence of setting values

    Return:
        Enumeration list of all possible combinations of keyword settings. Each configuration is a dict that maps
         kwarg -> setting
    """

    # validate kwargs
    kwargs_list = list(kwargs.keys())
    if len(kwargs_list) == 0:
        return []
    for kwarg in kwargs_list:
        if (
            not isinstance(kwargs[kwarg], list)
            and not isinstance(kwargs[kwarg], tuple)
            and not isinstance(kwargs[kwarg], np.ndarray)
        ):
            kwargs[kwarg] = [kwargs[kwarg]]
        if len(kwargs[kwarg]) == 0:
            raise ValueError(
                "Keyword "
                + kwarg
                + " cannot have zero values. Use explicit None or empty structure."
            )

    # make queue with first config
    first_kw = [{kwargs_list[0]: val} for val in kwargs[kwargs_list[0]]]
    if len(kwargs.keys()) == 1:
        return first_kw
    qu = queue.Queue()
    for config in first_kw:
        qu.put(config)

    # iteratively build configurations
    all_configs: List[Dict[str, Any]] = list()
    while not qu.empty():
        curr = qu.get()
        if len(curr.keys()) == len(kwargs.keys()):
            all_configs.append(curr)
        else:
            next_key = kwargs_list[len(curr.keys())]
            for val in kwargs[next_key]:
                new_dict = copy.deepcopy(curr)
                new_dict[next_key] = val
                qu.put(new_dict)
    return all_configs


def iter_params(*args, **kwargs) -> Callable:
    """
    Custom version of @parametrize function decorator that works with PyCharm. Can be used to parametrize external
    parameters to test the cross-product of values for input keyword argument parameters. See example usage.

    param **kwargs: Keyword arguments to function

    Return: New function that can be run in PyCharm IDE successfully or executed in pytest or nose test that loops
    over the possible combinations of parameters specified and runs each parameter combo. Test will pass only if all
    combinations pass and fail if any parameter combination fails.

    Example:

    @iter_params(a=[1, 2], b=[3], c=[1,4])
    def test_func(a: int, b: int, c: int) -> int:
        return a*b*c

    When test_func is executed, it will run with the configurations:
    [{a:1, b:3, c:1}, {a:2, b:3, c:1}, {a:1, b:3, c:4}, {a:2, b:3, c:4}]
    """

    def decorator(test_function: Callable) -> Callable:

        # wraps decorator is necessary so original function name is passed to nose tests / pytest.
        @wraps(test_function)
        def wrapper(*args1, **kwargs1) -> None:

            # loop over parameters of function
            all_settings = enumerate_settings(kwargs)
            for i, setting in enumerate(all_settings):

                # automatically run setUp() function of test class (if specified) before running function with each
                # parameter setting. Note that we can skip the first setUp() since UnitTest will run it natively.
                if len(args1) == 0:
                    raise ValueError(
                        "Using iter_params on a @static test function is not supported."
                    )
                elif type(args1[0]) is type:
                    raise ValueError(
                        "Using iter_params on a @classmethod test function is not supported."
                    )
                elif (i > 0) and hasattr(args1[0], "setUp"):
                    setup_func = getattr(args1[0], "setUp")
                    if callable(setup_func):
                        args1[0].setUp()

                # run function with setting
                test_function(*args1, **setting)

                # automatically run tearDown() function of test class (if specified) after running function with each
                # parameter setting. Note that we can skip the last tearDown() call since UnitTest will run it
                # natively.
                if (
                    i != (len(all_settings) - 1)
                    and len(args1) > 0
                    and type(args1[0]) is not type
                ):
                    if hasattr(args1[0], "tearDown"):
                        teardown_func = getattr(args1[0], "tearDown")
                        if callable(teardown_func):
                            args1[0].tearDown()

        # return wrapper
        return wrapper

    # return decorated test function runner
    return decorator
