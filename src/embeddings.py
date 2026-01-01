"""
Embedding models for RAG pipeline.
"""

import numpy as np
from openai import OpenAI
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIEmbeddings:
    """OpenAI embedding model."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Dimension depends on model
        self.dimension = 1536 if "3-small" in model else 3072
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                raise
        
        return np.array(embeddings)


class CachedEmbeddings:
    """
    Wrapper that caches embeddings to avoid redundant API calls.
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.cache = {}
        self.model_name = base_model.model_name
        self.dimension = base_model.dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text with caching."""
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.base_model.embed_text(text)
        self.cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch with caching."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                embeddings.append(None)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.base_model.embed_batch(texts_to_embed)
            
            # Update cache and results
            for text, embedding in zip(texts_to_embed, new_embeddings):
                self.cache[text] = embedding
            
            # Fill in results
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = embedding
        
        return np.array(embeddings)
    
    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            'cached_items': len(self.cache),
            'cache_size_mb': sum(emb.nbytes for emb in self.cache.values()) / (1024 * 1024)
        }

from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    """
    FREE local embeddings using sentence-transformers.
    No API costs! Runs on your machine.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence transformer.
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dim 
        - all-mpnet-base-v2: Better quality, 768 dim
        """
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Loaded (dimension: {self.dimension})")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )