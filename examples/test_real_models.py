"""
Test real OpenAI embeddings and generation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_eval_platform import Document, GroundTruthItem, RetrievalEvaluator, GenerationEvaluator
from src.advanced_rag_pipeline import AdvancedRAGPipeline

def main():
    print("="*70)
    print("TESTING REAL OPENAI MODELS")
    print("="*70)
    
    # Create test documents
    documents = [
        Document(
            doc_id="python",
            content="""
            Python was created by Guido van Rossum and first released in 1991.
            It emphasizes code readability and supports multiple programming paradigms.
            Python is used in web development, data science, AI, and automation.
            Popular frameworks include Django, Flask, NumPy, and TensorFlow.
            """
        ),
        Document(
            doc_id="javascript",
            content="""
            JavaScript was created by Brendan Eich in 1995 for Netscape Navigator.
            It's a high-level, dynamic language primarily used for web development.
            JavaScript runs in browsers and on servers (Node.js).
            Popular frameworks include React, Vue, Angular, and Express.
            """
        )
    ]
    
    print("\n1. Initializing RAG with real OpenAI models...")
    rag = AdvancedRAGPipeline(top_k=3)
    
    print("\n2. Ingesting documents...")
    chunks = rag.ingest_documents(documents, show_progress=True)
    
    print(f"\n3. Testing query with real LLM generation...")
    query = "Who created Python and when?"
    
    retrieval_result, generation_result = rag.query(query, verbose=True)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved chunks:")
    for i, (chunk, score) in enumerate(zip(retrieval_result.retrieved_chunks, retrieval_result.scores), 1):
        print(f"  {i}. {chunk.chunk_id} (score: {score:.3f})")
        print(f"     {chunk.content[:100].strip()}...")
    
    print(f"\nGenerated Answer:")
    print(f"  {generation_result.generated_answer}")
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)

if __name__ == "__main__":
    main()