from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from api.models import BatchProcessRequest, BatchJobStatus
from services.batch_processor import get_batch_processor

router = APIRouter(prefix="/api/batch", tags=["batch"])


@router.post("/process", response_model=Dict[str, str])
async def create_batch_job(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a batch processing job

    Args:
        request: Batch processing request
        background_tasks: FastAPI background tasks

    Returns:
        Job ID
    """
    try:
        processor = get_batch_processor()

        # Create job
        job_id = processor.create_job(
            input_folder=request.input_folder,
            output_folder=request.output_folder,
            label_folder=request.label_folder,
            prompts=request.prompts,
            confidence_threshold=request.confidence_threshold,
            process_videos=request.process_videos,
        )

        # Add processing to background tasks
        background_tasks.add_task(processor.process_job, job_id)

        return {
            "job_id": job_id,
            "message": "Batch processing started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create batch job: {str(e)}")


@router.get("/status/{job_id}", response_model=BatchJobStatus)
async def get_batch_status(job_id: str):
    """
    Get batch job status

    Args:
        job_id: Job ID

    Returns:
        Job status information
    """
    try:
        processor = get_batch_processor()
        status = processor.get_job_status(job_id)

        return status

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")
