"""
Semantic similarity calculation using Sentence-BERT
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from typing import List

# Global model instance (cached)
_model = None

def get_model() -> SentenceTransformer:
    """
    Get or load the Sentence-BERT model (singleton pattern)
    
    Returns:
        Loaded SentenceTransformer model
    """
    global _model
    
    if _model is None:
        model_name = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
        print(f"📥 Loading Sentence-BERT model: {model_name}")
        
        try:
            _model = SentenceTransformer(model_name)
            print(f"✅ Model loaded successfully! Embedding dimension: {_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    return _model

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using SBERT embeddings
    
    Args:
        text1: First text (e.g., resume)
        text2: Second text (e.g., job description)
    
    Returns:
        Similarity score between 0 and 1
    """
    try:
        model = get_model()
        
        # Generate embeddings for both texts
        embeddings = model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        # Ensure the score is between 0 and 1
        similarity = max(0.0, min(1.0, float(similarity)))
        
        return similarity
    
    except Exception as e:
        print(f"❌ Error calculating similarity: {str(e)}")
        raise Exception(f"Failed to calculate similarity: {str(e)}")

def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding vector for a single text
    
    Args:
        text: Input text
    
    Returns:
        Embedding vector as numpy array
    """
    model = get_model()
    return model.encode(text)

def batch_similarity(texts: List[str], reference_text: str) -> List[float]:
    """
    Calculate similarity of multiple texts against a reference text
    
    Args:
        texts: List of texts to compare
        reference_text: Reference text to compare against
    
    Returns:
        List of similarity scores
    """
    try:
        model = get_model()
        
        # Encode all texts at once (more efficient)
        all_texts = texts + [reference_text]
        embeddings = model.encode(all_texts, show_progress_bar=False)
        
        # Reference embedding is the last one
        ref_embedding = embeddings[-1].reshape(1, -1)
        
        # Calculate similarities for all texts
        similarities = []
        for i in range(len(texts)):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                ref_embedding
            )[0][0]
            similarities.append(max(0.0, min(1.0, float(sim))))
        
        return similarities
    
    except Exception as e:
        print(f"❌ Error in batch similarity: {str(e)}")
        raise Exception(f"Failed to calculate batch similarity: {str(e)}")

def calculate_semantic_score(resume_text: str, job_text: str) -> dict:
    """
    Calculate detailed semantic similarity metrics
    
    Args:
        resume_text: Resume text
        job_text: Job description text
    
    Returns:
        Dictionary with various similarity metrics
    """
    similarity = calculate_similarity(resume_text, job_text)
    
    # Convert to percentage
    percentage = similarity * 100
    
    # Categorize match quality
    if similarity >= 0.8:
        category = "Excellent Match"
    elif similarity >= 0.6:
        category = "Good Match"
    elif similarity >= 0.4:
        category = "Fair Match"
    else:
        category = "Poor Match"
    
    return {
        "similarity_score": similarity,
        "percentage": percentage,
        "category": category,
        "confidence": "high" if similarity >= 0.7 else "medium" if similarity >= 0.4 else "low"
    }