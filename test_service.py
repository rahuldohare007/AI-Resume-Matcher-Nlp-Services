"""
Test the NLP service endpoints
"""

import requests
import json
from typing import Dict

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_health():
    """Test health endpoint"""
    print_section("Testing Health Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"✅ Message: {data['message']}")
            print(f"✅ Version: {data['version']}")
        else:
            print(f"❌ Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")

def test_similarity():
    """Test similarity calculation"""
    print_section("Testing Similarity Calculation")
    
    payload = {
        "resume_text": """
        Experienced Python Developer with 5 years of expertise in machine learning 
        and data science. Proficient in TensorFlow, PyTorch, and scikit-learn. 
        Strong background in NLP, computer vision, and deep learning. Experience 
        with FastAPI, Django, and building scalable REST APIs. Familiar with AWS, 
        Docker, and CI/CD pipelines.
        """,
        "job_description": """
        We are looking for a Senior Machine Learning Engineer with strong Python 
        skills and experience in deep learning frameworks like TensorFlow and PyTorch. 
        The ideal candidate should have expertise in NLP and experience building 
        production ML systems. Knowledge of FastAPI, Docker, and cloud platforms 
        (AWS/GCP) is required.
        """
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/calculate-similarity",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Similarity Score: {data['similarity_score']:.2%}")
            print(f"\n📊 Matched Keywords ({len(data['matched_keywords'])}):")
            print(f"   {', '.join(data['matched_keywords'][:10])}")
            print(f"\n❌ Missing Keywords ({len(data['missing_keywords'])}):")
            print(f"   {', '.join(data['missing_keywords'][:10])}")
        else:
            print(f"❌ Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_batch_similarity():
    """Test batch similarity calculation"""
    print_section("Testing Batch Similarity")
    
    payload = {
        "resumes": [
            "Python developer with ML experience in TensorFlow and PyTorch",
            "Frontend developer skilled in React, TypeScript, and Next.js",
            "Data scientist with expertise in Python, R, and statistical analysis",
        ],
        "job_description": "Looking for ML engineer with Python and TensorFlow experience"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch-similarity",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Processed: {data['total_processed']} resumes")
            print("\n📊 Results (sorted by score):")
            for result in data['results']:
                print(f"   Resume #{result['index']}: {result['similarity_score']:.2%}")
        else:
            print(f"❌ Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Run all tests"""
    print("\n" + "🚀 " * 20)
    print("  RESUME MATCHER NLP SERVICE - API TESTS")
    print("🚀 " * 20)
    
    test_health()
    test_similarity()
    test_batch_similarity()
    
    print("\n" + "=" * 60)
    print("  ✅ All tests completed!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()