"""
Keyword extraction and matching
"""

import re
from collections import Counter
from typing import List, Tuple, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Comprehensive technical skills and keywords
TECHNICAL_KEYWORDS = {
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust',
    'php', 'swift', 'kotlin', 'scala', 'r', 'matlab',
    
    # Web Technologies
    'react', 'angular', 'vue', 'nextjs', 'next.js', 'nodejs', 'node.js', 'express',
    'django', 'flask', 'fastapi', 'spring', 'asp.net', 'html', 'css', 'sass',
    'tailwind', 'bootstrap', 'webpack', 'vite',
    
    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
    'dynamodb', 'oracle', 'sqlite', 'nosql', 'firebase',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
    'terraform', 'ansible', 'ci/cd', 'devops', 'linux', 'unix',
    
    # Data Science & ML
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'data science',
    'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy',
    'matplotlib', 'seaborn', 'jupyter', 'spark', 'hadoop', 'ai', 'artificial intelligence',
    
    # Mobile
    'android', 'ios', 'react native', 'flutter', 'xamarin',
    
    # Other Technologies
    'rest api', 'graphql', 'microservices', 'api', 'git', 'agile', 'scrum',
    'jira', 'testing', 'unit testing', 'selenium', 'jest', 'pytest',
    
    # Soft Skills
    'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
    'project management', 'collaboration'
}

def extract_keywords(text: str, top_n: int = 50) -> List[str]:
    """
    Extract important keywords from text
    
    Args:
        text: Input text
        top_n: Number of top keywords to return
    
    Returns:
        List of keywords sorted by importance
    """
    # Convert to lowercase
    text_lower = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text_lower)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Filter tokens: alphanumeric, not stopword, length > 2
    filtered_tokens = [
        token for token in tokens 
        if token.isalnum() 
        and token not in stop_words 
        and len(token) > 2
    ]
    
    # Find technical keywords in text
    found_technical = set()
    for keyword in TECHNICAL_KEYWORDS:
        # Handle multi-word keywords
        if ' ' in keyword:
            if keyword in text_lower:
                found_technical.add(keyword)
        else:
            if keyword in filtered_tokens:
                found_technical.add(keyword)
    
    # Count word frequencies
    word_freq = Counter(filtered_tokens)
    
    # Get top frequent words
    top_frequent = [word for word, freq in word_freq.most_common(top_n * 2)]
    
    # Combine technical keywords with frequent words
    all_keywords = list(found_technical) + [
        word for word in top_frequent 
        if word not in found_technical
    ]
    
    # Return top N unique keywords
    return all_keywords[:top_n]

def match_keywords(
    resume_keywords: List[str],
    job_keywords: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Match keywords between resume and job description
    
    Args:
        resume_keywords: Keywords from resume
        job_keywords: Keywords from job description
    
    Returns:
        Tuple of (matched_keywords, missing_keywords)
    """
    # Convert to lowercase sets for case-insensitive matching
    resume_set = set(kw.lower().strip() for kw in resume_keywords)
    job_set = set(kw.lower().strip() for kw in job_keywords)
    
    # Find matches and missing keywords
    matched = sorted(list(resume_set.intersection(job_set)))
    missing = sorted(list(job_set.difference(resume_set)))
    
    return matched, missing

def calculate_keyword_score(
    resume_keywords: List[str],
    job_keywords: List[str]
) -> float:
    """
    Calculate keyword match score (0-1)
    
    Args:
        resume_keywords: Keywords from resume
        job_keywords: Keywords from job description
    
    Returns:
        Match score between 0 and 1
    """
    if not job_keywords:
        return 0.0
    
    matched, _ = match_keywords(resume_keywords, job_keywords)
    
    score = len(matched) / len(job_keywords)
    return min(1.0, score)  # Cap at 1.0

def extract_skills(text: str) -> Set[str]:
    """
    Extract technical skills from text
    
    Args:
        text: Input text
    
    Returns:
        Set of identified skills
    """
    text_lower = text.lower()
    found_skills = set()
    
    for skill in TECHNICAL_KEYWORDS:
        if skill in text_lower:
            found_skills.add(skill)
    
    return found_skills

def get_keyword_analysis(resume_text: str, job_text: str) -> dict:
    """
    Get comprehensive keyword analysis
    
    Args:
        resume_text: Resume text
        job_text: Job description text
    
    Returns:
        Dictionary with keyword analysis metrics
    """
    resume_keywords = extract_keywords(resume_text, top_n=30)
    job_keywords = extract_keywords(job_text, top_n=30)
    
    matched, missing = match_keywords(resume_keywords, job_keywords)
    score = calculate_keyword_score(resume_keywords, job_keywords)
    
    return {
        "keyword_match_score": score,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "matched_keywords": matched[:15],
        "missing_keywords": missing[:15],
        "match_percentage": score * 100
    }