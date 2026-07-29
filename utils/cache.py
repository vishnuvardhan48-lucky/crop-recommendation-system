"""
Enterprise Cache Management
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import time
from collections import OrderedDict
import threading

from config.logging_config import get_logger

logger = get_logger(__name__)

class CacheManager:
    """
    Professional cache manager with LRU eviction and TTL
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
        
    def _generate_key(self, obj: Any) -> str:
        """Generate cache key from object"""
        if isinstance(obj, (dict, list)):
            obj_str = json.dumps(obj, sort_keys=True)
        else:
            obj_str = str(obj)
        return hashlib.md5(obj_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.timestamps[key] > self.ttl:
                    self.delete(key)
                    self.misses += 1
                    return None
                
                # Move to end (LRU)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                logger.debug(f"Cache hit: {key}")
                return value
            
            self.misses += 1
            logger.debug(f"Cache miss: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache"""
        with self.lock:
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            logger.debug(f"Cache set: {key}")
    
    def delete(self, key: str):
        """Delete item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]
            logger.debug(f"Cache deleted: {key}")
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            key, _ = self.cache.popitem(last=False)
            if key in self.timestamps:
                del self.timestamps[key]
            logger.debug(f"Evicted LRU item: {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(hit_rate, 2),
            'keys': list(self.cache.keys())
        }
    
    def get_many(self, keys: list) -> Dict[str, Any]:
        """Get multiple items from cache"""
        return {key: self.get(key) for key in keys}
    
    def set_many(self, items: Dict[str, Any]):
        """Set multiple items in cache"""
        for key, value in items.items():
            self.set(key, value)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache
    
    def get_or_set(self, key: str, func: callable, *args, **kwargs) -> Any:
        """Get from cache or set if not exists"""
        value = self.get(key)
        if value is None:
            value = func(*args, **kwargs)
            self.set(key, value)
        return value
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        value = self.get(key)
        if value is None:
            value = amount
        else:
            value += amount
        self.set(key, value)
        return value
    
    def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement counter in cache"""
        return self.increment(key, -amount)
    
    def cleanup_expired(self):
        """Remove expired items"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            for key in expired_keys:
                self.delete(key)
            logger.info(f"Cleaned up {len(expired_keys)} expired items")

class RedisCache:
    """Redis-based cache (if Redis is available)"""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        try:
            import redis
            self.client = redis.from_url(redis_url)
            self.default_ttl = default_ttl
            self.available = True
            logger.info("Redis cache initialized")
        except ImportError:
            logger.warning("Redis not available, using fallback cache")
            self.fallback = CacheManager()
            self.available = False
    
    def get(self, key: str) -> Optional[Any]:
        if self.available:
            value = self.client.get(key)
            return pickle.loads(value) if value else None
        return self.fallback.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if self.available:
            self.client.setex(
                key,
                ttl or self.default_ttl,
                pickle.dumps(value)
            )
        else:
            self.fallback.set(key, value, ttl)