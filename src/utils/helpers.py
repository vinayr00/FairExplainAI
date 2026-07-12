"""Helper functions for project utilities."""
import time

def timer_decorator(func):
    """Decorator to log function execution duration."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"Function '{func.__name__}' took {duration:.4f} seconds to execute.")
        return result
    return wrapper
