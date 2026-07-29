"""
Enterprise Logging Utilities
"""

import sys
import json
from pathlib import Path
from loguru import logger
from datetime import datetime

from config.settings import settings

class Logger:
    """Professional logging configuration"""
    
    def __init__(self):
        self._configure()
    
    def _configure(self):
        """Configure logging with multiple handlers"""
        
        # Remove default handler
        logger.remove()
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True,
            enqueue=True
        )
        
        # File handler with rotation
        log_file = settings.LOG_FILE
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        # JSON handler for structured logging
        logger.add(
            log_file.with_suffix('.json'),
            rotation=settings.LOG_ROTATION,
            format=lambda record: json.dumps({
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "module": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
                "context": record.get("extra", {})
            }) + "\n",
            level="INFO",
            enqueue=True
        )
        
        # Error file for critical errors
        logger.add(
            log_file.with_suffix('.error.log'),
            rotation="1 week",
            retention="3 months",
            level="ERROR",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        logger.success("Logging configured successfully")
    
    def get_logger(self):
        """Get configured logger"""
        return logger

# Global logger instance
log = Logger().get_logger()