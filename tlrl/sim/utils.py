"""
Utilities for working with RL+SUMO simulations.
"""

import time
import socket


def get_available_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def timing(num_runs: int = 1) -> callable:
    def decorator(f):
        """A decorator that prints the average execution time of a function."""

        def wrapper(*args, **kwargs):
            total_time = 0
            for _ in range(num_runs):
                start = time.time()
                result = f(*args, **kwargs)
                end = time.time()
                total_time += end - start
            avg_time = total_time / num_runs
            print(
                f"{f.__name__!r} executed in {avg_time:.3f} secs (avg of {num_runs} runs)"
            )
            return result

        return wrapper

    return decorator
