"""
Advanced RAG pipeline with real embeddings and LLM generation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_eval_platform import (
    Document, Chunk, RetrievalResult, GenerationResult,
    FixedSizeChunker, SimpleVectorStore
)
from src.embeddings import OpenAIEmbeddings, CachedEmbeddings
from src.generation import OpenAIGenerator
from typing import List, Tuple


class AdvancedRAGPipeline:
    """RAG pipeline with real embeddings and LLM generation."""
    
    def __init__(
        self,
        chunker: FixedSizeChunker = None,
        embedding_model = None,
        generator = None,
        vector_store: SimpleVectorStore = None,
        top_k: int = 5,
        use_cache: bool = True
    ):
        """
        Initialize advanced RAG pipeline.
        
        Args:
            chunker: Document chunker
            embedding_model: Embedding model (OpenAI, etc.)
            generator: LLM generator (OpenAI, Anthropic, etc.)
            vector_store: Vector store
            top_k: Number of chunks to retrieve
            use_cache: Whether to cache embeddings
        """
        self.chunker = chunker or FixedSizeChunker(chunk_size=500, overlap=100)
        
        # Initialize embedding model
        if embedding_model is None:
            base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.embedding_model = CachedEmbeddings(base_embeddings) if use_cache else base_embeddings
        else:
            self.embedding_model = embedding_model
        
        # Initialize generator
        self.generator = generator or OpenAIGenerator(model="gpt-4o-mini")
        
        self.vector_store = vector_store or SimpleVectorStore()
        self.top_k = top_k
    
    def ingest_documents(self, documents: List[Document], show_progress: bool = True):
        """Ingest and index documents."""
        all_chunks = []
        
        if show_progress:
            print(f"Chunking {len(documents)} documents...")
        
        # Chunk documents
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        if show_progress:
            print(f"Created {len(chunks)} chunks")
            print(f"Generating embeddings...")
        
        # Generate embeddings in batch
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_model.embed_batch(chunk_texts)
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to vector store
        self.vector_store.add_chunks(all_chunks)
        
        if show_progress:
            print(f"✓ Indexed {len(all_chunks)} chunks")
        
        return all_chunks
    
    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embedding_model.embed_text(query)
        chunks, scores = self.vector_store.search(query_embedding, self.top_k)
        
        return RetrievalResult(
            query=query,
            retrieved_chunks=chunks,
            scores=scores
        )
    
    def generate(self, query: str, retrieved_chunks: List[Chunk]) -> GenerationResult:
        """Generate answer using real LLM."""
        context_texts = [chunk.content for chunk in retrieved_chunks]
        generated_answer = self.generator.generate(query, context_texts)
        
        context = "\n\n".join([f"[{i+1}] {chunk.content}" for i, chunk in enumerate(retrieved_chunks)])
        
        return GenerationResult(
            query=query,
            context=context,
            generated_answer=generated_answer
        )
    
    def query(self, query: str, verbose: bool = False) -> Tuple[RetrievalResult, GenerationResult]:
        """End-to-end query pipeline."""
        if verbose:
            print(f"Query: {query}")
            print("Retrieving relevant chunks...")
        
        retrieval_result = self.retrieve(query)
        
        if verbose:
            print(f"Retrieved {len(retrieval_result.retrieved_chunks)} chunks")
            print("Generating answer...")
        
        generation_result = self.generate(query, retrieval_result.retrieved_chunks)
        
        if verbose:
            print("✓ Complete")
        
        return retrieval_result, generation_result