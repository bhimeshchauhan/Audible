"""
API route definitions for PDF processing.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..config import config
from ..services import process_pdf_to_graph_html

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================


class ProcessResponse(BaseModel):
    """Response model for successful PDF processing."""

    graphHtml: str


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    environment: str


# =============================================================================
# Routes
# =============================================================================


@router.post(
    "/process",
    response_model=ProcessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Processing failed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def process_pdf(file: UploadFile = File(...)):
    """
    Process an uploaded PDF file and return character relationship graph HTML.

    - **file**: PDF file to process (multipart/form-data)

    Returns JSON with `graphHtml` containing the interactive visualization HTML.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail={"error": "No filename provided", "detail": None},
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "detail": f"Expected PDF file, got: {file.filename}",
            },
        )

    # Validate content type if provided
    if file.content_type and file.content_type != "application/pdf":
        print(f"[WARN] Unexpected content type: {file.content_type}")

    # Save to temporary file
    temp_path = None
    try:
        # Read content first to check size
        content = await file.read()

        # Validate file size
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail={"error": "Empty file", "detail": "Uploaded PDF is empty"},
            )

        if len(content) > config.upload.max_size_bytes:
            max_mb = config.upload.max_size_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "File too large",
                    "detail": f"Maximum file size is {max_mb:.0f}MB",
                },
            )

        # Create temp file with .pdf extension
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            delete=False,
            dir=config.upload.temp_dir,
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(content)

        print(f"[INFO] Processing PDF: {file.filename} ({len(content)} bytes)")

        # Process the PDF
        graph_html = process_pdf_to_graph_html(temp_path)

        return ProcessResponse(graphHtml=graph_html)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "File not found", "detail": str(e)},
        )

    except ValueError as e:
        # Business logic errors (not enough characters, etc.)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Processing failed",
                "detail": str(e),
            },
        )

    except Exception as e:
        # Log the full traceback for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "detail": f"An unexpected error occurred: {type(e).__name__}",
            },
        )

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"[WARN] Failed to cleanup temp file: {cleanup_error}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", environment=config.environment)
