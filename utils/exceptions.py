"""
Enterprise Custom Exceptions
"""

class CropSystemException(Exception):
    """Base exception for crop system"""
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code or "CROP_ERROR"
        super().__init__(self.message)

class ModelNotFoundException(CropSystemException):
    """Raised when model is not found"""
    def __init__(self, path: str):
        super().__init__(
            f"Model not found at {path}",
            code="MODEL_NOT_FOUND"
        )

class DataValidationException(CropSystemException):
    """Raised when data validation fails"""
    def __init__(self, errors: list):
        super().__init__(
            f"Data validation failed: {', '.join(errors)}",
            code="DATA_VALIDATION_ERROR"
        )

class PredictionException(CropSystemException):
    """Raised when prediction fails"""
    def __init__(self, reason: str):
        super().__init__(
            f"Prediction failed: {reason}",
            code="PREDICTION_ERROR"
        )

class ConfigurationException(CropSystemException):
    """Raised when configuration is invalid"""
    def __init__(self, message: str):
        super().__init__(
            f"Configuration error: {message}",
            code="CONFIG_ERROR"
        )

class APIRateLimitException(CropSystemException):
    """Raised when API rate limit is exceeded"""
    def __init__(self, limit: int, reset_time: int):
        super().__init__(
            f"Rate limit of {limit} requests exceeded. Resets in {reset_time}s",
            code="RATE_LIMIT_EXCEEDED"
        )

class InvalidInputException(CropSystemException):
    """Raised when input validation fails"""
    def __init__(self, field: str, value: any, message: str):
        super().__init__(
            f"Invalid input for {field}: {value} - {message}",
            code="INVALID_INPUT"
        )