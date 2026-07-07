import io
import time
import logging
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

import sys
import os
# Ensure root directory is on Python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import HandwritingCNN
from src.ocr_pipeline import run_ocr_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OCR-API")

# Global variables for model and device to enable caching across requests
model = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages API startup and shutdown events.
    Loads the model weights once on startup to avoid expensive disk IO on each request.
    """
    global model, device
    logger.info("Initializing Handwriting OCR API...")
    
    # Path to the trained model weight file
    model_path = "models/ocr_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"Trained model checkpoint not found at: {model_path}")
        logger.error("Please run the training pipeline first using: python src/train.py")
        raise RuntimeError(f"Trained model not found at {model_path}")
        
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CNN model onto device: {device}")
        
        # Instantiate architecture and load state dict weights
        model = HandwritingCNN(num_classes=47)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Crucial: set to eval mode to disable dropout
        
        logger.info("CNN model loaded and cached successfully!")
    except Exception as e:
        logger.critical(f"Failed to load model during API startup: {e}")
        raise RuntimeError("Model initialization failure.") from e
        
    yield
    
    # Shutdown clean up
    logger.info("Shutting down Handwriting OCR API...")


# Initialize FastAPI application with lifespan caching
app = FastAPI(
    title="Handwriting OCR API",
    description="A production-quality REST API to perform handwriting text recognition using a custom PyTorch CNN model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", summary="Health Check Endpoint")
def read_root():
    """
    Verifies that the API is running and healthy.
    """
    return {"message": "Handwriting OCR API Running"}


@app.post("/predict", summary="Perform Handwritten Text OCR")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image of handwritten text to segment and recognize it.
    
    - **file**: Multipart file upload (must be a valid image file like PNG or JPEG).
    """
    # 1. Validate file content type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Rejected non-image upload attempt: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: '{file.content_type}'. Uploaded file must be an image."
        )

    try:
        # 2. Read raw file bytes
        contents = await file.read()
        
        # 3. Convert bytes to OpenCV image format
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode uploaded image bytes.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode the uploaded image. The file may be corrupted."
            )
            
        logger.info(f"Processing image upload: {file.filename} (shape={img.shape})")
        
        # 4. Execute the OCR pipeline and measure elapsed time
        start_time = time.time()
        recognized_text, _ = run_ocr_pipeline(
            image_path_or_np=img,
            model=model,
            device=device,
            save_vis=False
        )
        elapsed_time = time.time() - start_time
        
        logger.info("Inference completed successfully.")
        
        # 5. Return JSON response
        return {
            "text": recognized_text,
            "processing_time": f"{elapsed_time:.4f} seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error encountered during inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during character prediction: {str(e)}"
        )
