"""
Enterprise Metrics Collection
"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import json

from config.logging_config import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """
    Professional metrics collection for monitoring
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0
        self.predictions_by_crop = defaultdict(int)
        self.request_times = []
        self.endpoint_stats = defaultdict(lambda: {'count': 0, 'errors': 0, 'total_time': 0})
        self.daily_stats = defaultdict(lambda: {'predictions': 0, 'errors': 0})
        self.lock = threading.Lock()
        
    def track_request(self, endpoint: str, method: str, status_code: int, 
                     processing_time: float, **kwargs):
        """Track API request metrics"""
        with self.lock:
            self.request_count += 1
            self.total_response_time += processing_time
            self.request_times.append(processing_time)
            
            # Keep only last 1000 request times
            if len(self.request_times) > 1000:
                self.request_times.pop(0)
            
            # Endpoint stats
            stats = self.endpoint_stats[endpoint]
            stats['count'] += 1
            stats['total_time'] += processing_time
            
            if status_code >= 400:
                stats['errors'] += 1
                self.error_count += 1
            
            # Daily stats
            today = datetime.now().strftime('%Y-%m-%d')
            self.daily_stats[today]['predictions'] += 1
            if status_code >= 400:
                self.daily_stats[today]['errors'] += 1
    
    def track_prediction(self, crop: str, confidence: float):
        """Track prediction metrics"""
        with self.lock:
            self.predictions_by_crop[crop] += 1
    
    def track_error(self, endpoint: str, error_type: str):
        """Track error metrics"""
        with self.lock:
            self.error_count += 1
            logger.error(f"Error on {endpoint}: {error_type}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        with self.lock:
            avg_response_time = (
                self.total_response_time / self.request_count 
                if self.request_count > 0 else 0
            )
            
            p95_response_time = self._calculate_percentile(95)
            p99_response_time = self._calculate_percentile(99)
            
            return {
                'uptime_seconds': time.time() - self.start_time,
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'error_rate': (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
                'avg_response_time_ms': round(avg_response_time * 1000, 2),
                'p95_response_time_ms': round(p95_response_time * 1000, 2),
                'p99_response_time_ms': round(p99_response_time * 1000, 2),
                'predictions_by_crop': dict(self.predictions_by_crop),
                'endpoints': dict(self.endpoint_stats)
            }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        summary = self.get_summary()
        
        # Add daily stats
        summary['daily_stats'] = dict(self.daily_stats)
        
        # Add performance metrics
        summary['performance'] = {
            'requests_per_minute': self._calculate_rpm(),
            'peak_rpm': self._calculate_peak_rpm(),
            'error_trend': self._calculate_error_trend()
        }
        
        return summary
    
    def _calculate_percentile(self, percentile: int) -> float:
        """Calculate percentile of response times"""
        if not self.request_times:
            return 0
        
        sorted_times = sorted(self.request_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def _calculate_rpm(self) -> float:
        """Calculate requests per minute"""
        uptime_minutes = (time.time() - self.start_time) / 60
        return self.request_count / max(uptime_minutes, 1)
    
    def _calculate_peak_rpm(self) -> float:
        """Calculate peak requests per minute (last 5 minutes)"""
        recent_count = sum(
            1 for t in self.request_times[-300:]  # Last 5 minutes if 1 request per second
        )
        return recent_count / 5
    
    def _calculate_error_trend(self) -> str:
        """Calculate error trend"""
        if len(self.daily_stats) < 2:
            return "insufficient_data"
        
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        today_errors = self.daily_stats[today]['errors']
        yesterday_errors = self.daily_stats[yesterday]['errors']
        
        if yesterday_errors == 0:
            return "stable" if today_errors == 0 else "increasing"
        
        change = (today_errors - yesterday_errors) / yesterday_errors
        
        if change < -0.1:
            return "decreasing"
        elif change > 0.1:
            return "increasing"
        else:
            return "stable"
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.start_time = time.time()
            self.request_count = 0
            self.error_count = 0
            self.total_response_time = 0
            self.predictions_by_crop.clear()
            self.request_times.clear()
            self.endpoint_stats.clear()
            self.daily_stats.clear()
            logger.info("Metrics reset")

class PrometheusMetrics:
    """Prometheus integration for metrics"""
    
    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, Gauge
            self.available = True
            
            # Define metrics
            self.requests_total = Counter(
                'crop_requests_total',
                'Total number of requests',
                ['endpoint', 'method', 'status']
            )
            
            self.request_duration = Histogram(
                'crop_request_duration_seconds',
                'Request duration in seconds',
                ['endpoint']
            )
            
            self.predictions_total = Counter(
                'crop_predictions_total',
                'Total number of predictions',
                ['crop']
            )
            
            self.model_confidence = Gauge(
                'crop_model_confidence',
                'Model confidence score'
            )
            
            self.active_requests = Gauge(
                'crop_active_requests',
                'Number of active requests'
            )
            
            logger.info("Prometheus metrics initialized")
            
        except ImportError:
            logger.warning("Prometheus client not available")
            self.available = False
    
    def track_request(self, endpoint: str, method: str, status: int, duration: float):
        """Track request in Prometheus"""
        if self.available:
            self.requests_total.labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).inc()
            
            self.request_duration.labels(
                endpoint=endpoint
            ).observe(duration)
    
    def track_prediction(self, crop: str):
        """Track prediction in Prometheus"""
        if self.available:
            self.predictions_total.labels(crop=crop).inc()
    
    def set_model_confidence(self, confidence: float):
        """Set model confidence gauge"""
        if self.available:
            self.model_confidence.set(confidence)

# Global metrics collector
metrics_collector = MetricsCollector()
prometheus_metrics = PrometheusMetrics()