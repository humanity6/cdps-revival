"""
Standard response models for the API
"""
from pydantic import BaseModel
from typing import Any, Optional, Dict

class StandardResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None

class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Any] = None