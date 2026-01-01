"""
LLM generation for RAG pipeline.
"""

import os
from typing import List
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class OpenAIGenerator:
    """OpenAI LLM generator."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize OpenAI generator.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(self, query: str, context_chunks: List[str], max_tokens: int = 500) -> str:
        """
        Generate answer from query and context.
        
        Args:
            query: User's question
            context_chunks: List of relevant text chunks
            max_tokens: Maximum tokens in response
        """
        # Format context
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create prompt
        prompt = f"""Answer the question based ONLY on the provided context. If the answer cannot be found in the context, say "I cannot answer this based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0  # Deterministic for evaluation
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            raise


class AnthropicGenerator:
    """Anthropic Claude generator."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str = None):
        """
        Initialize Anthropic generator.
        
        Args:
            model: Anthropic model name
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found.")
        
        self.client = Anthropic(api_key=self.api_key)
    
    def generate(self, query: str, context_chunks: List[str], max_tokens: int = 500) -> str:
        """Generate answer from query and context."""
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""Answer the question based ONLY on the provided context. If the answer cannot be found in the context, say "I cannot answer this based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            raise