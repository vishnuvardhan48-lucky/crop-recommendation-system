"""
Enterprise API Middleware
"""

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, Optional
import time
import uuid
from datetime import datetime
import json

from config.logging_config import get_logger
from utils.metrics import metrics_collector
from utils.rate_limiter import RateLimiter

logger = get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")
        
        # Process request
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-MS"] = str(round(process_time, 2))
            
            # Log response
            logger.info(f"Response {request_id}: {response.status_code} ({process_time:.2f}ms)")
            
            # Track metrics
            metrics_collector.track_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                processing_time=process_time
            )
            
            return response
            
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            logger.error(f"Error {request_id}: {str(e)}")
            
            metrics_collector.track_error(
                endpoint=request.url.path,
                error_type=type(e).__name__
            )
            
            raise

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API authentication"""
    
    def __init__(self, app, api_keys: Optional[Dict[str, str]] = None):
        super().__init__(app)
        self.api_keys = api_keys or {}
        
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public endpoints
        public_paths = ['/', '/health', '/docs', '/redoc', '/openapi.json']
        if request.url.path in public_paths:
            return await call_next(request)
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return Response(
                content=json.dumps({"error": "API key required"}),
                status_code=401,
                media_type="application/json"
            )
        
        # Validate API key
        if api_key not in self.api_keys.values():
            return Response(
                content=json.dumps({"error": "Invalid API key"}),
                status_code=403,
                media_type="application/json"
            )
        
        # Add client info to request state
        client_name = [k for k, v in self.api_keys.items() if v == api_key][0]
        request.state.client = client_name
        
        return await call_next(request)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        
    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP or API key)
        client_id = request.headers.get("X-API-Key", request.client.host)
        
        # Check rate limit
        if not self.rate_limiter.check_limit(client_id):
            return Response(
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "limit": self.rate_limiter.limit,
                    "window": self.rate_limiter.window
                }),
                status_code=429,
                media_type="application/json"
            )
        
        return await call_next(request)

class CORSMiddleware(BaseHTTPMiddleware):
    """Middleware for CORS handling"""
    
    def __init__(self, app, allowed_origins: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        
    async def dispatch(self, request: Request, call_next):
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)
        
        # Add CORS headers
        origin = request.headers.get("origin")
        if origin and (self.allowed_origins == ["*"] or origin in self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
            response.headers["Access-Control-Max-Age"] = "3600"
        
        return response

class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # Only compress text responses
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith(("text/", "application/json")):
            return response
        
        # Compress response
        import gzip
        from io import BytesIO
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Compress
        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
            f.write(body)
        
        compressed = buffer.getvalue()
        
        # Return compressed response
        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics"""
    
    async def dispatch(self, request: Request, call_next):
        # Start timer
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        metrics_collector.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        return response

def setup_middleware(app):
    """Setup all middleware"""
    # Add middleware in order (last added, first executed)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(CompressionMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allowed_origins=["*"]
    )
    app.add_middleware(RequestLoggingMiddleware)
    
    # Rate limiting (commented by default)
    # rate_limiter = RateLimiter(limit=100, window=60)
    # app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    
    # Authentication (commented by default)
    # api_keys = {"client1": "key1", "client2": "key2"}
    # app.add_middleware(AuthenticationMiddleware, api_keys=api_keys)
    
    logger.info("Middleware setup complete")