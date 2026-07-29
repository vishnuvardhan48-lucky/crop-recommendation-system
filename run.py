#!/usr/bin/env python3
"""
Enterprise Crop Recommendation System
Main entry point with CLI interface
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional
import signal
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from config.logging_config import setup_logging, get_logger
from data.data_generator import DataGenerator
from data.data_loader import DataLoader
from models.train import ModelTrainer
from models.predict import PredictionService
from api.server import start_api_server
from web.app import run_web_app
from utils.health_check import HealthChecker
from utils.metrics import MetricsCollector

# Setup logging
setup_logging()
logger = get_logger(__name__)

class Application:
    """Main application controller"""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()
        self.prediction_service = None
        self.running = False
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        logger.info("Shutting down application...")
        self.running = False
        if self.prediction_service:
            self.prediction_service.cleanup()
        sys.exit(0)
        
    async def train_model(self, force: bool = False, optimize: bool = False):
        """Train the model"""
        logger.info("="*60)
        logger.info("🌱 TRAINING PIPELINE STARTED")
        logger.info("="*60)
        
        try:
            # Check if model exists and force flag
            if not force and settings.MODEL_PATH.exists():
                logger.warning("Model already exists. Use --force to retrain.")
                return None
                
            # Initialize components
            logger.info("📂 Initializing data components...")
            generator = DataGenerator(seed=settings.RANDOM_SEED)
            loader = DataLoader()
            
            # Load or generate data
            logger.info("📊 Loading data...")
            if settings.DATA_PATH.exists() and not force:
                df = loader.load_csv(settings.DATA_PATH)
                logger.info(f"✅ Loaded {len(df)} samples from {settings.DATA_PATH}")
            else:
                logger.info("🔄 Generating synthetic data...")
                df = generator.generate_dataset(settings.SAMPLES_PER_CROP)
                loader.save_csv(df, settings.DATA_PATH)
                logger.info(f"✅ Generated {len(df)} samples")
            
            # Initialize trainer
            logger.info("🤖 Initializing model trainer...")
            trainer = ModelTrainer()
            
            # Train model
            if optimize:
                logger.info("🎯 Running hyperparameter optimization...")
                metadata = trainer.train_with_optimization(df)
            else:
                logger.info("🎯 Training with default parameters...")
                metadata = trainer.train_pipeline(df)
            
            # Log results
            accuracy = metadata['evaluation']['metrics']['accuracy']
            logger.success(f"✅ Training complete! Accuracy: {accuracy:.2%}")
            
            # Save metadata
            trainer.save_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
            
    async def run_api_server(self):
        """Run the API server"""
        logger.info("="*60)
        logger.info("🌐 API SERVER STARTING")
        logger.info("="*60)
        
        try:
            await start_api_server(
                host=settings.API_HOST,
                port=settings.API_PORT,
                workers=settings.API_WORKERS
            )
        except Exception as e:
            logger.error(f"❌ API server failed: {e}")
            raise
            
    def run_web_app(self):
        """Run the web application"""
        logger.info("="*60)
        logger.info("🌐 WEB APPLICATION STARTING")
        logger.info("="*60)
        
        try:
            run_web_app(
                host=settings.WEB_HOST,
                port=settings.WEB_PORT
            )
        except Exception as e:
            logger.error(f"❌ Web app failed: {e}")
            raise
            
    async def run_batch_prediction(self, input_file: Path, output_file: Path):
        """Run batch prediction"""
        logger.info("="*60)
        logger.info("📊 BATCH PREDICTION STARTING")
        logger.info("="*60)
        
        try:
            # Load prediction service
            self.prediction_service = PredictionService()
            if not self.prediction_service.load_models():
                raise RuntimeError("Failed to load models")
                
            # Load input data
            loader = DataLoader()
            df = loader.load_csv(input_file)
            logger.info(f"✅ Loaded {len(df)} samples from {input_file}")
            
            # Make predictions
            results = self.prediction_service.predict_batch(df)
            logger.info(f"✅ Generated predictions for {len(results)} samples")
            
            # Save results
            loader.save_csv(results, output_file)
            logger.info(f"✅ Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {e}")
            raise
            
    async def run_health_check(self):
        """Run health check"""
        logger.info("="*60)
        logger.info("🏥 HEALTH CHECK")
        logger.info("="*60)
        
        status = await self.health_checker.check_all()
        
        if status['healthy']:
            logger.success("✅ All systems healthy")
        else:
            logger.error("❌ Health check failed")
            for service, info in status['services'].items():
                if not info['healthy']:
                    logger.error(f"   - {service}: {info.get('error', 'Unknown error')}")
        
        return status
        
    async def run(self):
        """Main async run method"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        self.setup_signal_handlers()
        
        if args.command == "train":
            await self.train_model(force=args.force, optimize=args.optimize)
            
        elif args.command == "api":
            await self.run_api_server()
            
        elif args.command == "web":
            self.run_web_app()
            
        elif args.command == "batch":
            await self.run_batch_prediction(
                Path(args.input),
                Path(args.output)
            )
            
        elif args.command == "health":
            await self.run_health_check()
            
        elif args.command == "all":
            # Train if needed, then run web app
            if not settings.MODEL_PATH.exists():
                await self.train_model()
            self.run_web_app()
            
        else:
            parser.print_help()
            
    def create_parser(self):
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Enterprise Crop Recommendation System",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Commands")
        
        # Train command
        train_parser = subparsers.add_parser("train", help="Train the model")
        train_parser.add_argument("--force", action="store_true", help="Force retraining")
        train_parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
        
        # API command
        api_parser = subparsers.add_parser("api", help="Run API server")
        
        # Web command
        web_parser = subparsers.add_parser("web", help="Run web application")
        
        # Batch prediction command
        batch_parser = subparsers.add_parser("batch", help="Run batch prediction")
        batch_parser.add_argument("--input", required=True, help="Input CSV file")
        batch_parser.add_argument("--output", required=True, help="Output CSV file")
        
        # Health check command
        health_parser = subparsers.add_parser("health", help="Run health check")
        
        # All command
        all_parser = subparsers.add_parser("all", help="Run complete system")
        
        return parser

def main():
    """Main entry point"""
    app = Application()
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()