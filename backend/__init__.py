"""
Backend package for PDF character relationship extraction API.

This package provides:
- FastAPI web server for PDF processing
- NLP services for character and relationship extraction
- Configuration management with environment variables
"""

from .config import config
from .main import app

__all__ = ["app", "config"]
