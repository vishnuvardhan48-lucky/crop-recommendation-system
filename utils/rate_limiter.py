"""
Enterprise Rate Limiter
"""

import time
from typing import Dict, Tuple
from collections import defaultdict
import threading

from config.logging_config import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """
    Professional rate limiter with sliding window
    """
    
    def __init__(self, limit: int = 100, window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            limit: Maximum number of requests per window
            window: Time window in seconds
        """
        self.limit = limit
        self.window = window
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
        
    def check_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit
        
        Args:
            client_id: Client identifier (IP or API key)
            
        Returns:
            True if within limit, False if exceeded
        """
        with self.lock:
            current_time = time.time()
            
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.window
            ]
            
            # Check limit
            if len(self.requests[client_id]) >= self.limit:
                logger.warning(f"Rate limit exceeded for {client_id}")
                return False
            
            # Add current request
            self.requests[client_id].append(current_time)
            return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client"""
        with self.lock:
            current_time = time.time()
            
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.window
            ]
            
            return max(0, self.limit - len(self.requests[client_id]))
    
    def get_reset_time(self, client_id: str) -> float:
        """Get time until rate limit resets"""
        with self.lock:
            if not self.requests[client_id]:
                return 0
            
            oldest = min(self.requests[client_id])
            return max(0, self.window - (time.time() - oldest))
    
    def get_stats(self, client_id: str) -> Dict[str, any]:
        """Get rate limit statistics for client"""
        remaining = self.get_remaining(client_id)
        reset_time = self.get_reset_time(client_id)
        
        return {
            'limit': self.limit,
            'remaining': remaining,
            'used': self.limit - remaining,
            'reset_in_seconds': reset_time,
            'window_seconds': self.window
        }
    
    def reset_client(self, client_id: str):
        """Reset rate limit for client"""
        with self.lock:
            if client_id in self.requests:
                del self.requests[client_id]
            logger.info(f"Rate limit reset for {client_id}")

class TokenBucket:
    """
    Token bucket algorithm for rate limiting
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens
        
        Returns:
            True if tokens were consumed, False if insufficient
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_available(self) -> float:
        """Get available tokens"""
        with self.lock:
            self._refill()
            return self.tokens

class DistributedRateLimiter:
    """Redis-based distributed rate limiter"""
    
    def __init__(self, redis_client, limit: int = 100, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window
    
    def check_limit(self, client_id: str) -> bool:
        """Check rate limit using Redis"""
        key = f"rate_limit:{client_id}"
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        now = time.time()
        window_start = now - self.window
        
        # Remove old requests
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, self.window)
        
        results = pipe.execute()
        current_count = results[1]
        
        return current_count < self.limit