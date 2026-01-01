"""
RAG Evaluation and Monitoring Platform
A production-ready system for evaluating and monitoring RAG pipelines.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
from collections import defaultdict

# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

@dataclass
class Document:
    """A document in the knowledge base."""
    doc_id: str
    content: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Chunk:
    """A chunk of a document."""
    chunk_id: str
    doc_id: str
    content: str
    start_idx: int
    end_idx: int
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        """Serialize chunk (excluding embedding for readability)."""
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'content': self.content,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx
        }

@dataclass
class GroundTruthItem:
    """A single ground truth evaluation example."""
    question: str
    relevant_chunk_ids: List[str]  # Ground truth chunks
    expected_answer: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RetrievalResult:
    """Result from retrieval step."""
    query: str
    retrieved_chunks: List[Chunk]
    scores: List[float]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class GenerationResult:
    """Result from generation step."""
    query: str
    context: str
    generated_answer: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class EvaluationMetrics:
    """Metrics from evaluation."""
    timestamp: str
    retrieval_metrics: Dict
    generation_metrics: Dict
    system_config: Dict
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# 2. CHUNKING
# ============================================================================

class FixedSizeChunker:
    """Simple fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into overlapping chunks."""
        chunks = []
        content = document.content
        start = 0
        chunk_idx = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            
            # Create unique chunk ID
            chunk_id = f"{document.doc_id}_chunk_{chunk_idx}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                content=chunk_content,
                start_idx=start,
                end_idx=end
            ))
            
            # Move forward with overlap
            start += self.chunk_size - self.overlap
            chunk_idx += 1
        
        return chunks
    
    def get_config_hash(self) -> str:
        """Get hash of chunking configuration for versioning."""
        config = f"fixed_{self.chunk_size}_{self.overlap}"
        return hashlib.md5(config.encode()).hexdigest()[:8]


# ============================================================================
# 3. EMBEDDING (Mock for demonstration)
# ============================================================================

class MockEmbeddingModel:
    """
    Mock embedding model for demonstration.
    Replace with OpenAI embeddings or other real model.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "mock-embeddings-v1"
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate mock embedding. Replace with real API call."""
        # Simple hash-based mock for reproducibility
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed_text(text) for text in texts])


# ============================================================================
# 4. VECTOR STORE (Simple In-Memory Implementation)
# ============================================================================

class SimpleVectorStore:
    """
    Simple in-memory vector store using numpy.
    For production, replace with FAISS or Pinecone.
    """
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_id_to_idx: Dict[str, int] = {}
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to the store."""
        start_idx = len(self.chunks)
        
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i
        
        # Stack embeddings
        new_embeddings = np.array([c.embedding for c in chunks])
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[Chunk], List[float]]:
        """Search for top-k similar chunks."""
        if self.embeddings is None or len(self.chunks) == 0:
            return [], []
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved_chunks = [self.chunks[idx] for idx in top_indices]
        scores = [float(similarities[idx]) for idx in top_indices]
        
        return retrieved_chunks, scores
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve chunk by ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None


# ============================================================================
# 5. RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline."""
    
    def __init__(
        self,
        chunker: FixedSizeChunker,
        embedding_model: MockEmbeddingModel,
        vector_store: SimpleVectorStore,
        top_k: int = 5
    ):
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
    
    def ingest_documents(self, documents: List[Document]):
        """Ingest and index documents."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            
            # Generate embeddings
            for chunk in chunks:
                chunk.embedding = self.embedding_model.embed_text(chunk.content)
            
            all_chunks.extend(chunks)
        
        self.vector_store.add_chunks(all_chunks)
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
        """
        Generate answer from retrieved context.
        This is a mock - replace with actual LLM call.
        """
        context = "\n\n".join([f"[{i+1}] {c.content}" for i, c in enumerate(retrieved_chunks)])
        
        # Mock generation - just return context summary
        generated_answer = f"Based on the provided context, here are the relevant passages about '{query}': {context[:200]}..."
        
        return GenerationResult(
            query=query,
            context=context,
            generated_answer=generated_answer
        )
    
    def query(self, query: str) -> Tuple[RetrievalResult, GenerationResult]:
        """End-to-end query pipeline."""
        retrieval_result = self.retrieve(query)
        generation_result = self.generate(query, retrieval_result.retrieved_chunks)
        return retrieval_result, generation_result


# ============================================================================
# 6. RETRIEVAL EVALUATION
# ============================================================================

class RetrievalEvaluator:
    """Evaluate retrieval quality against ground truth."""
    
    @staticmethod
    def recall_at_k(
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str],
        k: int
    ) -> float:
        """
        Recall@k: What fraction of relevant chunks are in top-k?
        """
        if not relevant_chunk_ids:
            return 0.0
        
        retrieved_set = set(retrieved_chunk_ids[:k])
        relevant_set = set(relevant_chunk_ids)
        
        hits = len(retrieved_set & relevant_set)
        return hits / len(relevant_set)
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str]
    ) -> float:
        """
        MRR: 1 / (rank of first relevant item)
        Returns 0 if no relevant items found.
        """
        relevant_set = set(relevant_chunk_ids)
        
        for rank, chunk_id in enumerate(retrieved_chunk_ids, 1):
            if chunk_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    def evaluate_batch(
        self,
        ground_truth: List[GroundTruthItem],
        rag_pipeline: RAGPipeline,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval on a batch of ground truth items.
        """
        recall_scores = {k: [] for k in k_values}
        mrr_scores = []
        
        for gt_item in ground_truth:
            retrieval_result = rag_pipeline.retrieve(gt_item.question)
            retrieved_ids = [c.chunk_id for c in retrieval_result.retrieved_chunks]
            
            # Calculate recall@k for each k
            for k in k_values:
                recall = self.recall_at_k(
                    retrieved_ids,
                    gt_item.relevant_chunk_ids,
                    k
                )
                recall_scores[k].append(recall)
            
            # Calculate MRR
            mrr = self.mean_reciprocal_rank(
                retrieved_ids,
                gt_item.relevant_chunk_ids
            )
            mrr_scores.append(mrr)
        
        # Aggregate metrics
        metrics = {
            'recall': {f'recall@{k}': np.mean(scores) for k, scores in recall_scores.items()},
            'mrr': np.mean(mrr_scores),
            'num_queries': len(ground_truth)
        }
        
        return metrics


# ============================================================================
# 7. GENERATION EVALUATION
# ============================================================================

class GenerationEvaluator:
    """Evaluate generation quality."""
    
    @staticmethod
    def faithfulness_score(generated_answer: str, context: str) -> float:
        """
        Simple faithfulness check: what fraction of answer is in context?
        For production, use LLM-based faithfulness scoring.
        """
        if not generated_answer or not context:
            return 0.0
        
        # Simple token overlap approach
        answer_tokens = set(generated_answer.lower().split())
        context_tokens = set(context.lower().split())
        
        if not answer_tokens:
            return 0.0
        
        overlap = len(answer_tokens & context_tokens)
        return overlap / len(answer_tokens)
    
    @staticmethod
    def relevance_score(generated_answer: str, question: str) -> float:
        """
        Simple relevance check: does answer contain question keywords?
        For production, use semantic similarity or LLM-based scoring.
        """
        if not generated_answer or not question:
            return 0.0
        
        question_keywords = set(question.lower().split())
        answer_tokens = set(generated_answer.lower().split())
        
        if not question_keywords:
            return 0.0
        
        overlap = len(question_keywords & answer_tokens)
        return overlap / len(question_keywords)
    
    def evaluate_batch(
        self,
        ground_truth: List[GroundTruthItem],
        rag_pipeline: RAGPipeline
    ) -> Dict:
        """Evaluate generation on a batch."""
        faithfulness_scores = []
        relevance_scores = []
        
        for gt_item in ground_truth:
            _, generation_result = rag_pipeline.query(gt_item.question)
            
            faith_score = self.faithfulness_score(
                generation_result.generated_answer,
                generation_result.context
            )
            faithfulness_scores.append(faith_score)
            
            rel_score = self.relevance_score(
                generation_result.generated_answer,
                gt_item.question
            )
            relevance_scores.append(rel_score)
        
        metrics = {
            'faithfulness': np.mean(faithfulness_scores),
            'relevance': np.mean(relevance_scores),
            'num_queries': len(ground_truth)
        }
        
        return metrics


# ============================================================================
# 8. MONITORING AND DRIFT DETECTION
# ============================================================================

class MetricsMonitor:
    """Track and monitor metrics over time."""
    
    def __init__(self, storage_path: str = "metrics_history.jsonl"):
        self.storage_path = Path(storage_path)
        self.history: List[EvaluationMetrics] = []
        self._load_history()
    
    def _load_history(self):
        """Load historical metrics."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.history.append(EvaluationMetrics(**data))
    
    def record_evaluation(self, metrics: EvaluationMetrics):
        """Record new evaluation metrics."""
        self.history.append(metrics)
        
        # Append to file
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def detect_drift(self, metric_name: str, threshold: float = 0.1) -> Dict:
        """
        Detect if a metric has drifted significantly.
        Compares recent average to historical baseline.
        """
        if len(self.history) < 10:
            return {'drift_detected': False, 'message': 'Insufficient history'}
        
        # Get last 10 evaluations
        recent = self.history[-10:]
        baseline = self.history[-30:-10] if len(self.history) >= 30 else self.history[:-10]
        
        def extract_metric(evals, name):
            values = []
            for e in evals:
                # Navigate nested dict structure
                if name in e.retrieval_metrics:
                    val = e.retrieval_metrics[name]
                    if isinstance(val, dict):
                        values.extend(val.values())
                    else:
                        values.append(val)
                elif name in e.generation_metrics:
                    values.append(e.generation_metrics[name])
            return values
        
        recent_values = extract_metric(recent, metric_name)
        baseline_values = extract_metric(baseline, metric_name)
        
        if not recent_values or not baseline_values:
            return {'drift_detected': False, 'message': 'Metric not found'}
        
        recent_mean = np.mean(recent_values)
        baseline_mean = np.mean(baseline_values)
        
        relative_change = abs(recent_mean - baseline_mean) / baseline_mean
        
        return {
            'drift_detected': relative_change > threshold,
            'metric': metric_name,
            'recent_mean': recent_mean,
            'baseline_mean': baseline_mean,
            'relative_change': relative_change,
            'threshold': threshold
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics of all metrics."""
        if not self.history:
            return {}
        
        summary = defaultdict(list)
        
        for eval_metrics in self.history:
            # Retrieval metrics
            for k, v in eval_metrics.retrieval_metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        summary[f'retrieval_{k}_{sub_k}'].append(sub_v)
                else:
                    summary[f'retrieval_{k}'].append(v)
            
            # Generation metrics
            for k, v in eval_metrics.generation_metrics.items():
                summary[f'generation_{k}'].append(v)
        
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1]
            }
            for metric, values in summary.items()
        }


# ============================================================================
# 9. EXAMPLE USAGE
# ============================================================================

def run_example():
    """Demonstration of the complete system."""
    
    print("=" * 60)
    print("RAG EVALUATION PLATFORM - DEMO")
    print("=" * 60)
    
    # Initialize components
    chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    embedding_model = MockEmbeddingModel(dimension=384)
    vector_store = SimpleVectorStore()
    rag = RAGPipeline(chunker, embedding_model, vector_store, top_k=5)
    
    # Create sample documents
    documents = [
        Document(
            doc_id="doc1",
            content="Python is a high-level programming language. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is known for its clear syntax and readability."
        ),
        Document(
            doc_id="doc2",
            content="Machine learning is a subset of artificial intelligence. It enables systems to learn and improve from experience without being explicitly programmed. Common algorithms include neural networks, decision trees, and support vector machines."
        ),
        Document(
            doc_id="doc3",
            content="Vector databases store data as high-dimensional vectors. They enable similarity search and are commonly used in RAG systems. Popular vector databases include Pinecone, Weaviate, and FAISS."
        )
    ]
    
    print("\n1. Ingesting documents...")
    chunks = rag.ingest_documents(documents)
    print(f"   Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Create ground truth
    ground_truth = [
        GroundTruthItem(
            question="What is Python?",
            relevant_chunk_ids=["doc1_chunk_0"],
            expected_answer="Python is a high-level programming language"
        ),
        GroundTruthItem(
            question="What are vector databases used for?",
            relevant_chunk_ids=["doc3_chunk_0"]
        )
    ]
    
    print("\n2. Running evaluation...")
    
    # Evaluate retrieval
    retrieval_evaluator = RetrievalEvaluator()
    retrieval_metrics = retrieval_evaluator.evaluate_batch(ground_truth, rag)
    print(f"\n   Retrieval Metrics:")
    print(f"   - Recall@5: {retrieval_metrics['recall']['recall@5']:.3f}")
    print(f"   - MRR: {retrieval_metrics['mrr']:.3f}")
    
    # Evaluate generation
    generation_evaluator = GenerationEvaluator()
    generation_metrics = generation_evaluator.evaluate_batch(ground_truth, rag)
    print(f"\n   Generation Metrics:")
    print(f"   - Faithfulness: {generation_metrics['faithfulness']:.3f}")
    print(f"   - Relevance: {generation_metrics['relevance']:.3f}")
    
    # Record metrics
    monitor = MetricsMonitor("demo_metrics.jsonl")
    evaluation = EvaluationMetrics(
        timestamp=datetime.now().isoformat(),
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        system_config={
            'chunk_size': chunker.chunk_size,
            'chunk_overlap': chunker.overlap,
            'top_k': rag.top_k,
            'embedding_model': embedding_model.model_name
        }
    )
    monitor.record_evaluation(evaluation)
    
    print("\n3. Testing query...")
    query = "Tell me about Python programming"
    retrieval_result, generation_result = rag.query(query)
    print(f"\n   Query: {query}")
    print(f"   Retrieved {len(retrieval_result.retrieved_chunks)} chunks")
    print(f"   Top chunk: {retrieval_result.retrieved_chunks[0].content[:100]}...")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check demo_metrics.jsonl for recorded metrics.")
    print("=" * 60)


if __name__ == "__main__":
    run_example()