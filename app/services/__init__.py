"""
NLP Services Module
"""

from .text_extractor import extract_text_from_file, clean_text
from .similarity import calculate_similarity, batch_similarity, get_model
from .keyword_matcher import extract_keywords, match_keywords, calculate_keyword_score

__all__ = [
    'extract_text_from_file',
    'clean_text',
    'calculate_similarity',
    'batch_similarity',
    'get_model',
    'extract_keywords',
    'match_keywords',
    'calculate_keyword_score'
]