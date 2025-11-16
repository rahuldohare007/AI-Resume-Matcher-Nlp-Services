"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class TextExtractionResponse(BaseModel):
    """Response model for text extraction"""
    success: bool
    text: str
    word_count: int
    char_count: int
    extracted_at: str

class SimilarityRequest(BaseModel):
    """Request model for similarity calculation"""
    resume_text: str = Field(..., min_length=10, description="Resume text content")
    job_description: str = Field(..., min_length=10, description="Job description text")

class SimilarityResponse(BaseModel):
    """Response model for similarity calculation"""
    success: bool
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    matched_keywords: List[str]
    missing_keywords: List[str]
    resume_keywords: List[str]
    job_keywords: List[str]

class BatchSimilarityRequest(BaseModel):
    """Request model for batch similarity calculation"""
    resumes: List[str] = Field(..., min_items=1, max_items=100)
    job_description: str = Field(..., min_length=10)

class BatchSimilarityResult(BaseModel):
    """Single result in batch similarity"""
    index: int
    similarity_score: float

class BatchSimilarityResponse(BaseModel):
    """Response model for batch similarity"""
    success: bool
    results: List[BatchSimilarityResult]
    total_processed: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: str
    version: str

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str