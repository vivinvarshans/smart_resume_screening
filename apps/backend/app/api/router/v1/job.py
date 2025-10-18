import logging
import traceback

from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, Depends, Request, status, Query
from fastapi.responses import JSONResponse

from app.core import get_db_session
from app.services import (
    JobService,
    JobServiceError,
    JobNotFoundError,
    JobValidationError,
    JobCreationError,
    JobProcessingError,
)
from app.schemas.pydantic.job import JobUploadRequest

job_router = APIRouter()
logger = logging.getLogger(__name__)


@job_router.post(
    "/upload",
    summary="stores the job posting in the database by parsing the JD into a structured format JSON",
)
async def upload_job(
    payload: JobUploadRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Accepts a job description as a MarkDown text and stores it in the database.
    """
    request_id = getattr(request.state, "request_id", str(uuid4()))

    allowed_content_types = [
        "application/json",
    ]

    content_type = request.headers.get("content-type")
    if not content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content-Type header is missing",
        )

    if content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Content-Type. Only {', '.join(allowed_content_types)} is/are allowed.",
        )

    try:
        job_service = JobService(db)
        request_data = payload.model_dump()
        logger.info(f"Processing job upload request: {request_data}")
        
        job_ids = await job_service.create_and_store_job(request_data)
        if not job_ids:
            logger.error("No job IDs were generated")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate job IDs",
            )
            
        logger.info(f"Successfully created jobs with IDs: {job_ids}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Job descriptions successfully processed",
                "request_id": request_id,
                "job_ids": job_ids
            }
        )

    except JobValidationError as e:
        logger.warning(f"Job validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    except JobCreationError as e:
        logger.error(f"Job creation error: {str(e)}\nTrace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job. Please try again.",
        )

    except JobProcessingError as e:
        logger.error(f"Job processing error: {str(e)}\nTrace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    except JobServiceError as e:
        logger.error(f"Job service error: {str(e)}\nTrace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the job."
        )

    except AssertionError as e:
        logger.error(f"Request validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        logger.error(f"Unexpected error during job upload: {str(e)}\nTrace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the job upload."
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Job data successfully processed",
            "job_ids": job_ids,
            "request": {
                "request_id": request_id,
                "payload": payload.model_dump(),
            },
        }
    )


@job_router.get(
    "",
    summary="Get job data from both job and processed_job models",
)
async def get_job(
    request: Request,
    job_id: str = Query(..., description="Job ID to fetch data for"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Retrieves job data from both job_model and processed_job model by job_id.

    Args:
        job_id: The ID of the job to retrieve

    Returns:
        Combined data from both job and processed_job models

    Raises:
        HTTPException: If the job is not found or if there's an error fetching data.
    """
    request_id = getattr(request.state, "request_id", str(uuid4()))
    headers = {"X-Request-ID": request_id}

    try:
        if not job_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_id is required",
            )

        job_service = JobService(db)
        job_data = await job_service.get_job_with_processed_data(
            job_id=job_id
        )
        
        if not job_data:
            raise JobNotFoundError(
                message=f"Job with id {job_id} not found"
            )

        return JSONResponse(
            content={
                "request_id": request_id,
                "data": job_data,
            },
            headers=headers,
        )
    
    except JobNotFoundError as e:
        logger.warning(f"Job not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
    except JobServiceError as e:
        logger.error(f"Job service error: {str(e)}\nTrace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching the job data."
        )
        
    except Exception as e:
        logger.error(f"Unexpected error fetching job: {str(e)}\nTrace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving job data.",
        )
