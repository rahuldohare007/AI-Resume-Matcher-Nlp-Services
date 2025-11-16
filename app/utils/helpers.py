"""
Helper utility functions
"""

import re
from typing import Optional

def format_score(score: float, as_percentage: bool = True) -> str:
    """
    Format similarity score for display
    
    Args:
        score: Similarity score (0-1)
        as_percentage: Return as percentage
    
    Returns:
        Formatted score string
    """
    if as_percentage:
        return f"{score * 100:.2f}%"
    return f"{score:.4f}"

def validate_text(text: str, min_length: int = 10) -> bool:
    """
    Validate text input
    
    Args:
        text: Text to validate
        min_length: Minimum required length
    
    Returns:
        True if valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    if len(text.strip()) < min_length:
        return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Replace spaces and special characters
    filename = re.sub(r'[^\w\-\.]', '_', filename)
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    return filename.lower()

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def extract_email(text: str) -> Optional[str]:
    """
    Extract email address from text
    
    Args:
        text: Text containing email
    
    Returns:
        Email address if found, None otherwise
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    
    if match:
        return match.group(0)
    
    return None

def extract_phone(text: str) -> Optional[str]:
    """
    Extract phone number from text
    
    Args:
        text: Text containing phone
    
    Returns:
        Phone number if found, None otherwise
    """
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    match = re.search(phone_pattern, text)
    
    if match:
        return match.group(0)
    
    return None