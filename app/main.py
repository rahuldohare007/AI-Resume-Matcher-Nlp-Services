"""
Main FastAPI application for Resume Matcher NLP Service
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
from datetime import datetime
import os
from dotenv import load_dotenv

from app.models import (
    TextExtractionResponse,
    SimilarityRequest,
    SimilarityResponse,
    BatchSimilarityRequest,
    BatchSimilarityResponse,
    BatchSimilarityResult,
    HealthResponse,
    ErrorResponse
)
from app.services.text_extractor import extract_text_from_file
from app.services.similarity import calculate_similarity, batch_similarity, get_model
from app.services.keyword_matcher import extract_keywords, match_keywords
from app import __version__

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Resume Matcher NLP Service",
    description="AI-powered resume and job description matching using Sentence-BERT",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Starting Resume Matcher NLP Service...")
    print(f"📌 Version: {__version__}")
    print(f"🌐 Allowed Origins: {allowed_origins}")
    print("📥 Loading Sentence-BERT model...")
    
    try:
        get_model()  # Pre-load and cache the model
        print("✅ NLP Service is ready!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down NLP Service...")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "message": "Resume Matcher NLP Service is running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": __version__
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "message": "All systems operational",
        "timestamp": datetime.utcnow().isoformat(),
        "version": __version__
    }

@app.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded PDF or DOCX file
    
    Parameters:
    - file: Upload file (PDF or DOCX)
    
    Returns:
    - Extracted text with metadata
    """
    try:
        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only PDF and DOCX files are allowed."
            )
        
        # Validate file size (10MB max)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds 10MB limit."
            )
        
        # Extract text
        extracted_text = extract_text_from_file(content, file.filename)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the file. Please ensure the file contains readable text."
            )
        
        return {
            "success": True,
            "text": extracted_text,
            "word_count": len(extracted_text.split()),
            "char_count": len(extracted_text),
            "extracted_at": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error extracting text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting text: {str(e)}"
        )

@app.post("/calculate-similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """
    Calculate semantic similarity between resume and job description
    
    Parameters:
    - resume_text: Text content of resume
    - job_description: Text content of job description
    
    Returns:
    - Similarity score (0-1) and keyword analysis
    """
    try:
        # Validate input
        if len(request.resume_text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume text is too short (minimum 10 characters)"
            )
        
        if len(request.job_description.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job description is too short (minimum 10 characters)"
            )
        
        # Calculate similarity score
        similarity_score = calculate_similarity(
            request.resume_text,
            request.job_description
        )
        
        # Extract keywords from both texts
        resume_keywords = extract_keywords(request.resume_text, top_n=30)
        job_keywords = extract_keywords(request.job_description, top_n=30)
        
        # Match keywords
        matched, missing = match_keywords(resume_keywords, job_keywords)
        
        return {
            "success": True,
            "similarity_score": float(similarity_score),
            "matched_keywords": matched[:20],  # Top 20 matched
            "missing_keywords": missing[:20],  # Top 20 missing
            "resume_keywords": resume_keywords[:20],  # Top 20 from resume
            "job_keywords": job_keywords[:20]  # Top 20 from job
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error calculating similarity: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating similarity: {str(e)}"
        )

@app.post("/batch-similarity", response_model=BatchSimilarityResponse)
async def batch_similarity_endpoint(request: BatchSimilarityRequest):
    """
    Calculate similarity for multiple resumes against one job description
    
    Parameters:
    - resumes: List of resume texts (max 100)
    - job_description: Job description text
    
    Returns:
    - List of similarity scores sorted by relevance
    """
    try:
        if len(request.resumes) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 resumes allowed per batch"
            )
        
        # Calculate similarities
        scores = batch_similarity(request.resumes, request.job_description)
        
        # Create results
        results = [
            BatchSimilarityResult(index=idx, similarity_score=score)
            for idx, score in enumerate(scores)
        ]
        
        # Sort by score descending
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return {
            "success": True,
            "results": results,
            "total_processed": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in batch processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch processing: {str(e)}"
        )

# For running directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "True").lower() == "true"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload
    )