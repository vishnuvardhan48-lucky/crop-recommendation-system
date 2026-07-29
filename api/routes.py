"""
Enterprise FastAPI Routes
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

from models.predict import prediction_service
from config.logging_config import get_logger
from utils.metrics import metrics_collector

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crop Recommendation API",
    description="Enterprise AI-Powered Crop Recommendation System",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class PredictionRequest(BaseModel):
    """Prediction request model"""
    N: float = Field(..., ge=0, le=200, description="Nitrogen content (kg/ha)")
    P: float = Field(..., ge=0, le=200, description="Phosphorus content (kg/ha)")
    K: float = Field(..., ge=0, le=200, description="Potassium content (kg/ha)")
    temperature: float = Field(..., ge=0, le=50, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=300, description="Rainfall (mm)")
    
    class Config:
        schema_extra = {
            "example": {
                "N": 90,
                "P": 45,
                "K": 40,
                "temperature": 25,
                "humidity": 70,
                "ph": 6.5,
                "rainfall": 150
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    data: List[PredictionRequest] = Field(..., description="List of prediction requests")
    
    @validator('data')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100')
        return v

class PredictionResponse(BaseModel):
    """Prediction response model"""
    success: bool
    primary_recommendation: Optional[str] = None
    confidence: Optional[float] = None
    top_3: Optional[List[Dict[str, Any]]] = None
    all_probabilities: Optional[Dict[str, float]] = None
    warnings: List[str] = []
    processing_time_ms: float
    request_id: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    model_loaded: bool
    uptime: float
    predictions_count: int
    metrics: Dict[str, Any]

# Middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time header"""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    response.headers["X-Process-Time-MS"] = str(round(process_time, 2))
    return response

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return """
    <html>
        <head>
            <title>Crop Recommendation API</title>
            <style>
                body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #2E7D32; }
                .endpoint { background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; }
                code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🌱 Crop Recommendation API</h1>
            <p>Enterprise AI-Powered Crop Recommendation System</p>
            
            <div class="endpoint">
                <h3>📊 Available Endpoints:</h3>
                <ul>
                    <li><code>GET /health</code> - Health check</li>
                    <li><code>GET /info</code> - Model information</li>
                    <li><code>POST /predict</code> - Single prediction</li>
                    <li><code>POST /predict/batch</code> - Batch predictions</li>
                    <li><code>GET /crops</code> - List supported crops</li>
                    <li><code>GET /metrics</code> - System metrics</li>
                </ul>
            </div>
            
            <div class="endpoint">
                <h3>📚 Documentation:</h3>
                <ul>
                    <li><a href="/api/docs">Swagger UI</a></li>
                    <li><a href="/api/redoc">ReDoc</a></li>
                </ul>
            </div>
            
            <p>Version: 3.0.0</p>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        model_loaded=prediction_service.is_loaded,
        uptime=metrics_collector.get_uptime(),
        predictions_count=prediction_service.prediction_count,
        metrics=metrics_collector.get_summary()
    )

@app.get("/info")
async def model_info():
    """Get model information"""
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return prediction_service.get_model_info()

@app.get("/crops")
async def list_crops():
    """List all supported crops"""
    if not prediction_service.is_loaded or not prediction_service.class_names:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "count": len(prediction_service.class_names),
        "crops": prediction_service.class_names
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a single prediction"""
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Make prediction
        result = prediction_service.predict(
            N=request.N,
            P=request.P,
            K=request.K,
            temperature=request.temperature,
            humidity=request.humidity,
            ph=request.ph,
            rainfall=request.rainfall
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Track metrics in background
        background_tasks.add_task(
            metrics_collector.track_request,
            endpoint="/predict",
            status="success",
            processing_time=processing_time
        )
        
        return PredictionResponse(
            success=True,
            primary_recommendation=result.get('primary_recommendation'),
            confidence=result.get('confidence'),
            top_3=result.get('top_3'),
            all_probabilities=result.get('all_probabilities'),
            warnings=result.get('warnings', []),
            processing_time_ms=processing_time,
            request_id=request_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        background_tasks.add_task(
            metrics_collector.track_request,
            endpoint="/predict",
            status="error",
            processing_time=processing_time
        )
        
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Make batch predictions"""
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    results = []
    
    try:
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame([r.dict() for r in request.data])
        
        # Make batch predictions
        predictions = prediction_service.predict_batch(df)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Format results
        for idx, row in predictions.iterrows():
            results.append({
                "input": request.data[idx].dict(),
                "prediction": row['predicted_crop'],
                "confidence": row['confidence']
            })
        
        # Track metrics
        background_tasks.add_task(
            metrics_collector.track_request,
            endpoint="/predict/batch",
            status="success",
            processing_time=processing_time,
            batch_size=len(request.data)
        )
        
        return {
            "success": True,
            "results": results,
            "processing_time_ms": processing_time,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        background_tasks.add_task(
            metrics_collector.track_request,
            endpoint="/predict/batch",
            status="error",
            processing_time=processing_time
        )
        
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return metrics_collector.get_detailed_metrics()

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting API server...")
    
    # Load model
    if not prediction_service.load_models():
        logger.warning("Model not loaded. Some endpoints may not work.")
    
    logger.success("API server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down API server...")
    prediction_service.cleanup()
    logger.success("API server shutdown complete")