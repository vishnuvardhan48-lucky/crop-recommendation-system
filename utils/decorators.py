"""
Enterprise Decorators
"""

import functools
import time
from typing import Any, Callable
from datetime import datetime

from utils.logger import log
from utils.metrics import metrics_collector
from utils.exceptions import CropSystemException

def timer(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            log.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            log.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper

def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            log.info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            log.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        log.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise
                    log.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying...")
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

def memoize(func: Callable) -> Callable:
    """Decorator to memoize function results"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper

def validate_input(schema: dict):
    """Decorator to validate function input"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Simplified validation - in production use Pydantic
            for param, rules in schema.items():
                if param in kwargs:
                    value = kwargs[param]
                    if 'min' in rules and value < rules['min']:
                        raise ValueError(f"{param} must be >= {rules['min']}")
                    if 'max' in rules and value > rules['max']:
                        raise ValueError(f"{param} must be <= {rules['max']}")
                    if 'type' in rules and not isinstance(value, rules['type']):
                        raise ValueError(f"{param} must be of type {rules['type']}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            metrics_collector.track_request(
                endpoint=func.__name__,
                method="function",
                status_code=200,
                processing_time=elapsed
            )
            return result
        except Exception as e:
            elapsed = time.time() - start
            metrics_collector.track_error(
                endpoint=func.__name__,
                error_type=type(e).__name__
            )
            raise
    return wrapper