"""
My First RAG Evaluation
A simple example using your own documents.
"""

import sys
from pathlib import Path

# Add parent directory to Python path (automatically finds it!)
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_eval_platform import (
    Document, GroundTruthItem, FixedSizeChunker,
    MockEmbeddingModel, SimpleVectorStore, RAGPipeline,
    RetrievalEvaluator, GenerationEvaluator
)

def main():
    print("="*70)
    print("MY FIRST RAG EVALUATION")
    print("="*70)
    
    # Step 1: Create your documents
    print("\n1. Creating documents...")
    documents = [
        Document(
            doc_id="python_doc",
            content="""
            Python is a high-level, interpreted programming language created by 
            Guido van Rossum. First released in 1991, Python emphasizes code 
            readability with its use of significant indentation. It supports 
            multiple programming paradigms including procedural, object-oriented, 
            and functional programming. Python is widely used in web development, 
            data science, artificial intelligence, scientific computing, and 
            automation. Popular frameworks include Django, Flask, NumPy, and 
            TensorFlow.
            """
        ),
        Document(
            doc_id="ml_doc",
            content="""
            Machine learning is a branch of artificial intelligence that enables 
            computers to learn from data without being explicitly programmed. 
            Common algorithms include neural networks, decision trees, support 
            vector machines, and random forests. Machine learning is categorized 
            into supervised learning, unsupervised learning, and reinforcement 
            learning. Applications include image recognition, natural language 
            processing, recommendation systems, fraud detection, and autonomous 
            vehicles.
            """
        ),
        Document(
            doc_id="vector_db_doc",
            content="""
            Vector databases store data as high-dimensional vectors, which are 
            mathematical representations of data. They enable similarity search 
            and are commonly used in RAG (Retrieval-Augmented Generation) systems. 
            Popular vector databases include Pinecone, Weaviate, Milvus, and FAISS. 
            These databases use algorithms like HNSW, IVF, and ANN for efficient 
            nearest neighbor search. Vector databases are essential for semantic 
            search, recommendation engines, and AI applications.
            """
        )
    ]
    
    # Step 2: Set up RAG pipeline
    print("2. Setting up RAG pipeline...")
    chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    embedding_model = MockEmbeddingModel(dimension=384)
    vector_store = SimpleVectorStore()
    rag = RAGPipeline(chunker, embedding_model, vector_store, top_k=5)
    
    # Step 3: Ingest documents
    print("3. Ingesting documents...")
    chunks = rag.ingest_documents(documents)
    print(f"   ✓ Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Step 4: Show all chunks (so you can create ground truth)
    print("\n4. Your chunks:")
    print("-" * 70)
    for i, chunk in enumerate(chunks):
        print(f"\n📄 Chunk #{i+1}")
        print(f"   ID: {chunk.chunk_id}")
        print(f"   From: {chunk.doc_id}")
        print(f"   Content: {chunk.content[:120].strip()}...")
        print("-" * 70)
    
    # Step 5: Create ground truth
    print("\n5. Creating ground truth...")
    ground_truth = [
        GroundTruthItem(
            question="Who created Python?",
            relevant_chunk_ids=["python_doc_chunk_0"]
        ),
        GroundTruthItem(
            question="What are some machine learning algorithms?",
            relevant_chunk_ids=["ml_doc_chunk_0"]
        ),
        GroundTruthItem(
            question="What are vector databases used for?",
            relevant_chunk_ids=["vector_db_doc_chunk_0"]
        )
    ]
    print(f"   ✓ Created {len(ground_truth)} test questions")
    
    # Step 6: Evaluate retrieval
    print("\n6. Evaluating retrieval quality...")
    retrieval_eval = RetrievalEvaluator()
    ret_metrics = retrieval_eval.evaluate_batch(ground_truth, rag)
    
    # Step 7: Evaluate generation
    print("7. Evaluating generation quality...")
    generation_eval = GenerationEvaluator()
    gen_metrics = generation_eval.evaluate_batch(ground_truth, rag)
    
    # Step 8: Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📊 Retrieval Metrics:")
    print(f"   Recall@1:  {ret_metrics['recall']['recall@1']:.1%}")
    print(f"   Recall@3:  {ret_metrics['recall']['recall@3']:.1%}")
    print(f"   Recall@5:  {ret_metrics['recall']['recall@5']:.1%}")
    print(f"   Recall@10: {ret_metrics['recall']['recall@10']:.1%}")
    print(f"   MRR:       {ret_metrics['mrr']:.3f}")
    
    print("\n📝 Generation Metrics:")
    print(f"   Faithfulness: {gen_metrics['faithfulness']:.1%}")
    print(f"   Relevance:    {gen_metrics['relevance']:.1%}")
    
    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("="*70)
    
    # Step 9: Test a query
    print("\n8. Testing live query...")
    test_query = "Tell me about Python programming"
    print(f"   Query: '{test_query}'")
    
    retrieval_result, generation_result = rag.query(test_query)
    
    print(f"\n   Retrieved {len(retrieval_result.retrieved_chunks)} chunks:")
    for i, chunk in enumerate(retrieval_result.retrieved_chunks[:3], 1):
        print(f"   {i}. {chunk.chunk_id} (score: {retrieval_result.scores[i-1]:.3f})")
    
    print(f"\n   Generated answer preview:")
    print(f"   {generation_result.generated_answer[:150]}...")

if __name__ == "__main__":
    main()