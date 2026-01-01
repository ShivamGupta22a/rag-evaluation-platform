"""
Vector store implementations: FAISS, Simple in-memory.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
from src.rag_eval_platform import Chunk
import pickle
from pathlib import Path


class FAISSVectorStore:
    """
    FAISS vector store for efficient similarity search.
    Used by: Meta, OpenAI, Pinecone internally.
    """
    
    def __init__(self, dimension: int = 384, index_type: str = "Flat"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: "Flat" (exact) or "IVF" (approximate, faster)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: dict = {}
        
        # Create FAISS index
        if index_type == "Flat":
            # Exact search (small datasets)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        elif index_type == "IVF":
            # Approximate search (large datasets)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        print(f"✓ FAISS {index_type} index created (dimension: {dimension})")
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to FAISS index."""
        if not chunks:
            return
        
        start_idx = len(self.chunks)
        
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed (IVF only)
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i
        
        print(f"✓ Added {len(chunks)} chunks to FAISS index")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[Chunk], List[float]]:
        """Search for similar chunks."""
        if len(self.chunks) == 0:
            return [], []
        
        # Normalize query
        query_emb = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
        
        # Get chunks
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        retrieved_scores = scores[0].tolist()
        
        return retrieved_chunks, retrieved_scores
    
    def save(self, path: str):
        """Save index and chunks to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save chunks
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.chunk_id_to_idx, f)
        
        print(f"✓ Saved FAISS index to {path}")
    
    def load(self, path: str):
        """Load index and chunks from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load chunks
        with open(path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(path / "metadata.pkl", 'rb') as f:
            self.chunk_id_to_idx = pickle.load(f)
        
        print(f"✓ Loaded FAISS index from {path}")
    
    def get_stats(self):
        """Get index statistics."""
        return {
            'total_chunks': len(self.chunks),
            'index_type': self.index_type,
            'dimension': self.dimension,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }