"""
Enterprise Logging Configuration
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from pythonjsonlogger import jsonlogger
from datetime import datetime

from .settings import settings

class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru"""
    
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
            
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
            
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

class JSONFormatter:
    """Custom JSON formatter for structured logging"""
    
    def __call__(self, record: Dict[str, Any]) -> str:
        record['timestamp'] = datetime.utcnow().isoformat()
        record['environment'] = settings.ENVIRONMENT
        record['app_version'] = settings.APP_VERSION
        return json.dumps(record)

def setup_logging():
    """Configure logging for the application"""
    
    # Remove default handlers
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
    
    if settings.LOG_FORMAT == 'json':
        logger.add(
            log_file,
            format=JSONFormatter(),
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            compression="gz",
            level=settings.LOG_LEVEL,
            enqueue=True,
            serialize=True
        )
    else:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            compression="gz",
            level=settings.LOG_LEVEL,
            enqueue=True
        )
    
    # Error file for critical errors only
    logger.add(
        settings.LOG_FILE.with_suffix('.error.log'),
        format="{time} | {level} | {message}",
        rotation="1 week",
        retention="3 months",
        level="ERROR",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set third-party loggers to WARNING
    for logger_name in [
        'urllib3', 'asyncio', 'aiohttp', 'boto3', 'botocore',
        's3transfer', 'requests', 'werkzeug', 'matplotlib'
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger.success("Logging configured successfully")

def get_logger(name: str):
    """Get a logger instance with context"""
    return logger.bind(module=name)

class LoggerMixin:
    """Mixin class to add logging capability"""
    
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger