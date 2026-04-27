from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .routes import segmentation, batch, export
from services.storage import get_storage_service

# Create FastAPI app
app = FastAPI(
    title="SAM 3 Labeling Tool API",
    description="Backend API for SAM 3-powered image and video labeling",
    version="1.0.0"
)

# CORS middleware - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(segmentation.router)
app.include_router(batch.router)
app.include_router(export.router)

# Expose generated output images for simple browser-based testing.
storage = get_storage_service()
app.mount("/outputs", StaticFiles(directory=str(storage.outputs_path)), name="outputs")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SAM 3 Labeling Tool API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
