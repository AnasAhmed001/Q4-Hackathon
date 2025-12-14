from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ...services.ingestion_service import ingestion_service
from ...utils.logging import log_api_call, app_logger
import time


router = APIRouter()


class IngestRequest(BaseModel):
    file_path: str
    check_duplicates: bool = True


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_count: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class IngestDirectoryRequest(BaseModel):
    directory_path: str
    recursive: bool = True


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """
    Ingest endpoint with duplicate detection and progress tracking
    """
    start_time = time.time()

    try:
        # Log the API call
        log_api_call(
            endpoint="/api/v1/ingest",
            method="POST"
        )

        # Process the ingestion
        result = ingestion_service.ingest_file(
            file_path=request.file_path,
            check_duplicates=request.check_duplicates
        )

        # Calculate response time
        response_time = time.time() - start_time
        log_api_call(
            endpoint="/api/v1/ingest",
            method="POST",
            response_time=response_time * 1000,  # Convert to milliseconds
            status_code=200
        )

        return IngestResponse(
            status=result['status'],
            message=result['message'],
            chunks_count=result.get('chunks_count'),
            details=result
        )
    except Exception as e:
        app_logger.error(f"Error in ingest endpoint: {str(e)}")
        log_api_call(
            endpoint="/api/v1/ingest",
            method="POST",
            status_code=500
        )
        raise HTTPException(status_code=500, detail=f"Error processing ingestion request: {str(e)}")


@router.post("/ingest/directory")
async def ingest_directory_endpoint(request: IngestDirectoryRequest):
    """
    Ingest all files in a directory
    """
    start_time = time.time()

    try:
        # Log the API call
        log_api_call(
            endpoint="/api/v1/ingest/directory",
            method="POST"
        )

        # Process the directory ingestion
        result = ingestion_service.ingest_directory(
            directory_path=request.directory_path,
            recursive=request.recursive
        )

        # Calculate response time
        response_time = time.time() - start_time
        log_api_call(
            endpoint="/api/v1/ingest/directory",
            method="POST",
            response_time=response_time * 1000,  # Convert to milliseconds
            status_code=200
        )

        return result
    except Exception as e:
        app_logger.error(f"Error in ingest directory endpoint: {str(e)}")
        log_api_call(
            endpoint="/api/v1/ingest/directory",
            method="POST",
            status_code=500
        )
        raise HTTPException(status_code=500, detail=f"Error processing directory ingestion: {str(e)}")


@router.post("/ingest/docs")
async def ingest_docs_directory_endpoint():
    """
    Run initial ingestion of all website/docs/ content to populate Qdrant collection
    """
    start_time = time.time()

    try:
        # Log the API call
        log_api_call(
            endpoint="/api/v1/ingest/docs",
            method="POST"
        )

        # Process the docs directory ingestion
        result = ingestion_service.ingest_from_docs_directory()

        # Calculate response time
        response_time = time.time() - start_time
        log_api_call(
            endpoint="/api/v1/ingest/docs",
            method="POST",
            response_time=response_time * 1000,  # Convert to milliseconds
            status_code=200
        )

        return result
    except Exception as e:
        app_logger.error(f"Error in ingest docs directory endpoint: {str(e)}")
        log_api_call(
            endpoint="/api/v1/ingest/docs",
            method="POST",
            status_code=500
        )
        raise HTTPException(status_code=500, detail=f"Error processing docs directory ingestion: {str(e)}")


@router.post("/ingest/file-upload")
async def ingest_file_upload_endpoint(file: UploadFile = File(...)):
    """
    Upload and ingest a single file
    """
    start_time = time.time()

    try:
        # Log the API call
        log_api_call(
            endpoint="/api/v1/ingest/file-upload",
            method="POST"
        )

        # Save uploaded file temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding='utf-8') as temp_file:
            content = await file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process the ingestion
            result = ingestion_service.ingest_file(
                file_path=temp_file_path,
                check_duplicates=True
            )

            # Calculate response time
            response_time = time.time() - start_time
            log_api_call(
                endpoint="/api/v1/ingest/file-upload",
                method="POST",
                response_time=response_time * 1000,  # Convert to milliseconds
                status_code=200
            )

            return IngestResponse(
                status=result['status'],
                message=result['message'],
                chunks_count=result.get('chunks_count'),
                details=result
            )
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    except Exception as e:
        app_logger.error(f"Error in ingest file upload endpoint: {str(e)}")
        log_api_call(
            endpoint="/api/v1/ingest/file-upload",
            method="POST",
            status_code=500
        )
        raise HTTPException(status_code=500, detail=f"Error processing file upload ingestion: {str(e)}")