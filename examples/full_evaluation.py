"""
Full evaluation with real models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_eval_platform import (
    Document, GroundTruthItem, RetrievalEvaluator,
    GenerationEvaluator, MetricsMonitor, EvaluationMetrics
)
from src.advanced_rag_pipeline import AdvancedRAGPipeline
from datetime import datetime

def main():
    print("="*70)
    print("FULL RAG EVALUATION WITH REAL MODELS")
    print("="*70)
    
    # Your documents
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
            automation. Popular frameworks include Django, Flask, NumPy, Pandas,
            and TensorFlow.
            """
        ),
        Document(
            doc_id="javascript_doc",
            content="""
            JavaScript was created by Brendan Eich in 1995 while he was working
            at Netscape Communications. It was designed in just 10 days to enable
            interactive web pages. JavaScript is a high-level, dynamic, weakly typed
            language that conforms to the ECMAScript specification. It's primarily
            used for client-side web development but also runs on servers via Node.js.
            Popular frameworks and libraries include React, Vue, Angular, and Express.
            """
        ),
        Document(
            doc_id="ml_doc",
            content="""
            Machine learning is a branch of artificial intelligence that enables 
            computers to learn from data without being explicitly programmed. 
            Common algorithms include neural networks, decision trees, support 
            vector machines, random forests, and gradient boosting. Machine learning 
            is categorized into supervised learning, unsupervised learning, and 
            reinforcement learning. Applications include image recognition, natural 
            language processing, recommendation systems, fraud detection, and 
            autonomous vehicles.
            """
        )
    ]
    
    # Ground truth
    ground_truth = [
        GroundTruthItem(
            question="Who created Python and when was it first released?",
            relevant_chunk_ids=["python_doc_chunk_0"]
        ),
        GroundTruthItem(
            question="What company was Brendan Eich working for when he created JavaScript?",
            relevant_chunk_ids=["javascript_doc_chunk_0"]
        ),
        GroundTruthItem(
            question="What are some common machine learning algorithms?",
            relevant_chunk_ids=["ml_doc_chunk_0"]
        ),
        GroundTruthItem(
            question="What frameworks are popular for Python web development?",
            relevant_chunk_ids=["python_doc_chunk_0"]
        )
    ]
    
    print("\n1. Setting up RAG pipeline with real models...")
    rag = AdvancedRAGPipeline(top_k=5)
    
    print("\n2. Ingesting documents...")
    chunks = rag.ingest_documents(documents, show_progress=True)
    
    print(f"\n3. Running evaluation on {len(ground_truth)} test questions...")
    
    # Evaluate retrieval
    print("\n   Evaluating retrieval...")
    retrieval_eval = RetrievalEvaluator()
    ret_metrics = retrieval_eval.evaluate_batch(ground_truth, rag)
    
    # Evaluate generation
    print("   Evaluating generation...")
    generation_eval = GenerationEvaluator()
    gen_metrics = generation_eval.evaluate_batch(ground_truth, rag)
    
    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📊 Retrieval Metrics:")
    print(f"   Recall@1:  {ret_metrics['recall']['recall@1']:.1%}")
    print(f"   Recall@3:  {ret_metrics['recall']['recall@3']:.1%}")
    print(f"   Recall@5:  {ret_metrics['recall']['recall@5']:.1%}")
    print(f"   MRR:       {ret_metrics['mrr']:.3f}")
    
    print("\n📝 Generation Metrics:")
    print(f"   Faithfulness: {gen_metrics['faithfulness']:.1%}")
    print(f"   Relevance:    {gen_metrics['relevance']:.1%}")
    
    # Save metrics
    print("\n4. Saving metrics...")
    monitor = MetricsMonitor("evaluation_metrics.jsonl")
    evaluation = EvaluationMetrics(
        timestamp=datetime.now().isoformat(),
        retrieval_metrics=ret_metrics,
        generation_metrics=gen_metrics,
        system_config={
            'embedding_model': rag.embedding_model.model_name,
            'generator_model': rag.generator.model_name,
            'chunk_size': rag.chunker.chunk_size,
            'chunk_overlap': rag.chunker.overlap,
            'top_k': rag.top_k
        }
    )
    monitor.record_evaluation(evaluation)
    
    print("\n" + "="*70)
    print("✅ Evaluation complete! Metrics saved to evaluation_metrics.jsonl")
    print("="*70)
    
    # Show a sample query
    print("\n5. Testing live query...")
    test_query = "What is Python used for?"
    print(f"   Query: '{test_query}'")
    
    _, gen_result = rag.query(test_query)
    print(f"\n   Answer: {gen_result.generated_answer}")

if __name__ == "__main__":
    main()