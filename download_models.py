"""
Download required NLP models
Run this once after installing requirements: python download_models.py
"""

import nltk
from sentence_transformers import SentenceTransformer
import os

def download_nltk_data():
    """Download required NLTK datasets"""
    print("📥 Downloading NLTK data...")
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            print(f"  ✅ {dataset}")
        except Exception as e:
            print(f"  ❌ {dataset}: {str(e)}")

def download_sbert_model():
    """Download Sentence-BERT model"""
    print("\n📥 Downloading Sentence-BERT model...")
    print("⏳ This may take a few minutes (downloading ~80MB)...\n")
    
    try:
        # This will download and cache the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence-BERT model downloaded successfully!")
        
        # Test the model
        test_embedding = model.encode("This is a test sentence.")
        print(f"✅ Model test successful! Embedding dimension: {len(test_embedding)}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        raise

def main():
    """Main function to download all required models"""
    print("=" * 60)
    print("🚀 Downloading NLP Models for Resume Matcher")
    print("=" * 60)
    
    download_nltk_data()
    download_sbert_model()
    
    print("\n" + "=" * 60)
    print("✅ All models downloaded successfully!")
    print("=" * 60)
    print("\n🎯 You can now run the service with:")
    print("   python -m uvicorn app.main:app --reload --port 8000")

if __name__ == "__main__":
    main()