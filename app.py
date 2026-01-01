"""
Streamlit Dashboard for RAG Evaluation Platform
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.rag_eval_platform import Document, GroundTruthItem
from src.embeddings import SentenceTransformerEmbeddings, CachedEmbeddings
from src.vector_stores import FAISSVectorStore
from src.advanced_rag_pipeline import AdvancedRAGPipeline
from src.generation import OpenAIGenerator
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="RAG Evaluation Platform",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 RAG Evaluation & Monitoring Platform")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Embedding model selection
    embedding_choice = st.selectbox(
        "Embedding Model",
        ["SentenceTransformer (FREE)", "OpenAI (Paid)"],
        help="SentenceTransformer runs locally with no API costs"
    )
    
    # Generation model
    use_generation = st.checkbox("Enable LLM Generation", value=False)
    
    if use_generation:
        st.warning("⚠️ LLM generation uses OpenAI credits")
        generator_model = st.selectbox(
            "Generator Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        )
    
    # RAG parameters
    st.subheader("RAG Parameters")
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top K Results", 1, 10, 5, 1)
    
    st.markdown("---")
    st.caption("Built with Streamlit, FAISS, and Sentence Transformers")

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Ingestion", "🔍 Query", "📊 Evaluation", "📈 Metrics History"])

# Tab 1: Document Ingestion
with tab1:
    st.header("Document Ingestion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add Documents")
        
        # Text input
        doc_text = st.text_area(
            "Paste your document text",
            height=200,
            placeholder="Enter document content here..."
        )
        
        doc_id = st.text_input("Document ID", placeholder="e.g., doc_001")
        
        if st.button("Add Document", type="primary"):
            if doc_text and doc_id:
                # Initialize pipeline if needed
                if st.session_state.rag_pipeline is None:
                    with st.spinner("Initializing RAG pipeline..."):
                        from src.rag_eval_platform import FixedSizeChunker
                        
                        # Choose embedding model
                        if "FREE" in embedding_choice:
                            base_emb = SentenceTransformerEmbeddings()
                            emb_model = CachedEmbeddings(base_emb)
                        else:
                            from src.embeddings import OpenAIEmbeddings
                            base_emb = OpenAIEmbeddings()
                            emb_model = CachedEmbeddings(base_emb)
                        
                        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
                        vector_store = FAISSVectorStore(dimension=emb_model.dimension)
                        
                        # Generator (optional)
                        generator = None
                        if use_generation:
                            generator = OpenAIGenerator(model=generator_model)
                        
                        st.session_state.rag_pipeline = AdvancedRAGPipeline(
                            chunker=chunker,
                            embedding_model=emb_model,
                            generator=generator,
                            vector_store=vector_store,
                            top_k=top_k
                        )
                
                # Ingest document
                with st.spinner("Processing document..."):
                    doc = Document(doc_id=doc_id, content=doc_text)
                    chunks = st.session_state.rag_pipeline.ingest_documents([doc], show_progress=False)
                    st.session_state.indexed = True
                
                st.success(f"✅ Document '{doc_id}' processed! Created {len(chunks)} chunks.")
            else:
                st.error("Please provide both document text and ID")
    
    with col2:
        st.subheader("Pipeline Status")
        
        if st.session_state.rag_pipeline:
            st.success("✅ Pipeline initialized")
            
            # Show stats
            if hasattr(st.session_state.rag_pipeline.vector_store, 'get_stats'):
                stats = st.session_state.rag_pipeline.vector_store.get_stats()
                st.metric("Total Chunks", stats['total_chunks'])
                st.metric("Vector Dimension", stats['dimension'])
                st.info(f"Index: {stats['index_type']}")
        else:
            st.info("Pipeline not initialized yet")

# Tab 2: Query
with tab2:
    st.header("Query Documents")
    
    if not st.session_state.indexed:
        st.warning("⚠️ Please index some documents first (Tab 1)")
    else:
        query = st.text_input("Enter your question:", placeholder="What is Python used for?")
        
        if st.button("Search", type="primary") and query:
            with st.spinner("Searching..."):
                retrieval_result, generation_result = st.session_state.rag_pipeline.query(query)
            
            st.subheader("🔍 Retrieved Chunks")
            for i, (chunk, score) in enumerate(zip(retrieval_result.retrieved_chunks, retrieval_result.scores), 1):
                with st.expander(f"Chunk {i} - {chunk.chunk_id} (Score: {score:.3f})"):
                    st.text(chunk.content)
            
            if use_generation and generation_result:
                st.subheader("💬 Generated Answer")
                st.info(generation_result.generated_answer)

# Tab 3: Evaluation
with tab3:
    st.header("Evaluation")
    
    if not st.session_state.indexed:
        st.warning("⚠️ Please index some documents first")
    else:
        st.subheader("Ground Truth Examples")
        
        # Simple ground truth input
        num_examples = st.number_input("Number of test questions", 1, 10, 3)
        
        ground_truth = []
        for i in range(num_examples):
            with st.expander(f"Example {i+1}"):
                question = st.text_input(f"Question {i+1}", key=f"q_{i}")
                chunk_ids = st.text_input(f"Relevant Chunk IDs (comma-separated)", key=f"c_{i}")
                
                if question and chunk_ids:
                    ground_truth.append(GroundTruthItem(
                        question=question,
                        relevant_chunk_ids=[c.strip() for c in chunk_ids.split(",")]
                    ))
        
        if st.button("Run Evaluation") and ground_truth:
            with st.spinner("Evaluating..."):
                from src.rag_eval_platform import RetrievalEvaluator, GenerationEvaluator
                
                ret_eval = RetrievalEvaluator()
                ret_metrics = ret_eval.evaluate_batch(ground_truth, st.session_state.rag_pipeline)
                
                gen_eval = GenerationEvaluator()
                gen_metrics = gen_eval.evaluate_batch(ground_truth, st.session_state.rag_pipeline)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Retrieval Metrics")
                st.metric("Recall@1", f"{ret_metrics['recall']['recall@1']:.1%}")
                st.metric("Recall@5", f"{ret_metrics['recall']['recall@5']:.1%}")
                st.metric("MRR", f"{ret_metrics['mrr']:.3f}")
            
            with col2:
                st.subheader("📝 Generation Metrics")
                st.metric("Faithfulness", f"{gen_metrics['faithfulness']:.1%}")
                st.metric("Relevance", f"{gen_metrics['relevance']:.1%}")

# Tab 4: Metrics History
with tab4:
    st.header("Metrics Over Time")
    
    # Load metrics from file
    metrics_file = Path("evaluation_metrics.jsonl")
    
    if metrics_file.exists():
        # Read metrics
        metrics_data = []
        with open(metrics_file, 'r') as f:
            for line in f:
                metrics_data.append(json.loads(line))
        
        if metrics_data:
            # Extract data for plotting
            timestamps = [m['timestamp'] for m in metrics_data]
            recall_5 = [m['retrieval_metrics']['recall']['recall@5'] for m in metrics_data]
            mrr = [m['retrieval_metrics']['mrr'] for m in metrics_data]
            faithfulness = [m['generation_metrics']['faithfulness'] for m in metrics_data]
            
            # Create plots
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=recall_5,
                mode='lines+markers',
                name='Recall@5',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=mrr,
                mode='lines+markers',
                name='MRR',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=faithfulness,
                mode='lines+markers',
                name='Faithfulness',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Metrics Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Score",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Recall@5", f"{sum(recall_5)/len(recall_5):.1%}")
            with col2:
                st.metric("Avg MRR", f"{sum(mrr)/len(mrr):.3f}")
            with col3:
                st.metric("Avg Faithfulness", f"{sum(faithfulness)/len(faithfulness):.1%}")
        else:
            st.info("No metrics data yet")
    else:
        st.info("No metrics file found. Run evaluations to generate data.")

# Footer
st.markdown("---")
st.caption("RAG Evaluation Platform | Built with Streamlit, FAISS, Sentence Transformers, and OpenAI")