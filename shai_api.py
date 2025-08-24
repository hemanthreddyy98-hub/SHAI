#!/usr/bin/env python3
"""
SHAI API Server - Real AI Integration
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import logging

# FastAPI for web server
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# AI Models
import openai
import google.generativeai as genai
import anthropic
import cohere
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Advanced NLP
import spacy
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import nltk

# Vector Database
import chromadb
from chromadb.config import Settings

# Utilities
import numpy as np
from rich.console import Console
import structlog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    model_used: str
    processing_time: float
    confidence: float
    analysis: Dict[str, Any]

class SHAIAPI:
    def __init__(self):
        self.app = FastAPI(title="SHAI API", version="1.0.0")
        self.setup_cors()
        self.setup_routes()
        
        # Initialize AI models
        self.initialize_models()
        
        # Initialize NLP components
        self.initialize_nlp()
        
        # Initialize knowledge base
        self.initialize_knowledge_base()
        
        # Model performance tracking
        self.model_performance = {}
        
    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            return await self.process_chat(request)
        
        @self.app.post("/api/chat/stream")
        async def chat_stream_endpoint(request: ChatRequest):
            return StreamingResponse(
                self.stream_chat(request),
                media_type="text/plain"
            )
        
        @self.app.get("/api/models")
        async def get_models():
            return {
                "available_models": list(self.models.keys()),
                "performance_stats": self.model_performance
            }
        
        @self.app.post("/api/knowledge/add")
        async def add_knowledge(content: str, metadata: Dict = None):
            return await self.add_to_knowledge_base(content, metadata)
        
        @self.app.get("/api/knowledge/search")
        async def search_knowledge(query: str, limit: int = 5):
            return await self.search_knowledge_base(query, limit)
    
    def initialize_models(self):
        """Initialize all available AI models"""
        self.models = {}
        
        # OpenAI Models
        if os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.models['gpt-4'] = {
                'type': 'openai',
                'model': 'gpt-4',
                'max_tokens': 8000
            }
            self.models['gpt-3.5-turbo'] = {
                'type': 'openai',
                'model': 'gpt-3.5-turbo',
                'max_tokens': 4000
            }
        
        # Google Gemini Models
        if os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.models['gemini-pro'] = {
                'type': 'gemini',
                'model': 'gemini-1.5-pro',
                'max_tokens': 1000000
            }
            self.models['gemini-flash'] = {
                'type': 'gemini',
                'model': 'gemini-1.5-flash',
                'max_tokens': 1000000
            }
        
        # Anthropic Claude Models
        if os.getenv('ANTHROPIC_API_KEY'):
            self.claude_client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
            self.models['claude-3-opus'] = {
                'type': 'claude',
                'model': 'claude-3-opus-20240229',
                'max_tokens': 200000
            }
            self.models['claude-3-sonnet'] = {
                'type': 'claude',
                'model': 'claude-3-sonnet-20240229',
                'max_tokens': 200000
            }
            self.models['claude-3-haiku'] = {
                'type': 'claude',
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 200000
            }
        
        # Cohere Model
        if os.getenv('COHERE_API_KEY'):
            self.cohere_client = cohere.Client(
                api_key=os.getenv('COHERE_API_KEY')
            )
            self.models['cohere-command'] = {
                'type': 'cohere',
                'model': 'command',
                'max_tokens': 4000
            }
        
        # Local Models (if available)
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.models['local-sentiment'] = {
                'type': 'local',
                'model': 'sentiment-analysis',
                'max_tokens': 1000
            }
        except:
            pass
        
        console.print(f"[green]Initialized {len(self.models)} AI models[/green]")
    
    def initialize_nlp(self):
        """Initialize NLP components"""
        try:
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                console.print("[yellow]spaCy model not found, installing...[/yellow]")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            console.print("[green]NLP components initialized[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing NLP: {e}[/red]")
    
    def initialize_knowledge_base(self):
        """Initialize vector knowledge base"""
        try:
            self.kb_client = chromadb.PersistentClient(
                path="./shai_knowledge",
                settings=Settings(anonymized_telemetry=False)
            )
            self.kb_collection = self.kb_client.get_or_create_collection("shai_knowledge")
            console.print("[green]Knowledge base initialized[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing knowledge base: {e}[/red]")
    
    async def select_best_model(self, messages: List[Message], task_type: str = None) -> str:
        """Intelligently select the best model for the task"""
        if not messages:
            return list(self.models.keys())[0]
        
        last_message = messages[-1].content.lower()
        
        # Analyze message complexity and type
        word_count = len(last_message.split())
        has_code = any(keyword in last_message for keyword in ['code', 'program', 'function', 'algorithm', 'debug'])
        is_creative = any(keyword in last_message for keyword in ['write', 'create', 'story', 'poem', 'creative'])
        is_analytical = any(keyword in last_message for keyword in ['analyze', 'compare', 'evaluate', 'study', 'research'])
        is_technical = any(keyword in last_message for keyword in ['technical', 'engineering', 'system', 'architecture'])
        
        # Model selection logic
        if is_creative and word_count > 50:
            return 'gpt-4' if 'gpt-4' in self.models else 'claude-3-opus'
        elif is_analytical and word_count > 100:
            return 'claude-3-opus' if 'claude-3-opus' in self.models else 'gpt-4'
        elif is_technical and has_code:
            return 'claude-3-sonnet' if 'claude-3-sonnet' in self.models else 'gpt-4'
        elif word_count > 200:
            return 'gemini-pro' if 'gemini-pro' in self.models else 'claude-3-opus'
        elif word_count < 20:
            return 'gemini-flash' if 'gemini-flash' in self.models else 'gpt-3.5-turbo'
        else:
            return 'gpt-4' if 'gpt-4' in self.models else list(self.models.keys())[0]
    
    async def call_ai_model(self, model_name: str, messages: List[Message], **kwargs) -> Dict[str, Any]:
        """Call the specified AI model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model_config = self.models[model_name]
        start_time = datetime.now()
        
        try:
            if model_config['type'] == 'openai':
                response = await openai.ChatCompletion.acreate(
                    model=model_config['model'],
                    messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 2000)
                )
                return {
                    'content': response.choices[0].message.content,
                    'model': model_name,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'usage': response.usage
                }
            
            elif model_config['type'] == 'gemini':
                model = genai.GenerativeModel(model_config['model'])
                prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
                response = await model.generate_content_async(prompt)
                return {
                    'content': response.text,
                    'model': model_name,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            elif model_config['type'] == 'claude':
                response = await self.claude_client.messages.create(
                    model=model_config['model'],
                    messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                    max_tokens=kwargs.get('max_tokens', 2000),
                    temperature=kwargs.get('temperature', 0.7)
                )
                return {
                    'content': response.content[0].text,
                    'model': model_name,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            elif model_config['type'] == 'cohere':
                response = await self.cohere_client.chat(
                    message=messages[-1].content,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 2000)
                )
                return {
                    'content': response.text,
                    'model': model_name,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")
        
        except Exception as e:
            logger.error(f"Error calling model {model_name}: {e}")
            raise
    
    async def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze message for sentiment, entities, and complexity"""
        try:
            # Sentiment analysis
            blob = TextBlob(message)
            sentiment = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
            # Entity recognition
            doc = self.nlp(message)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Complexity analysis
            word_count = len(message.split())
            sentence_count = len(nltk.sent_tokenize(message))
            avg_word_length = np.mean([len(word) for word in message.split()]) if word_count > 0 else 0
            
            return {
                'sentiment': sentiment,
                'entities': entities,
                'complexity': {
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'avg_word_length': avg_word_length,
                    'complexity_score': (word_count * avg_word_length) / max(sentence_count, 1)
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing message: {e}")
            return {}
    
    async def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict]:
        """Search knowledge base for relevant information"""
        try:
            results = self.kb_collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            return [
                {
                    'content': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    async def add_to_knowledge_base(self, content: str, metadata: Dict = None) -> Dict:
        """Add content to knowledge base"""
        try:
            import uuid
            doc_id = str(uuid.uuid4())
            
            self.kb_collection.add(
                documents=[content],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            
            return {
                'id': doc_id,
                'status': 'added',
                'content_length': len(content)
            }
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {e}")
            raise
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request and return intelligent response"""
        start_time = datetime.now()
        
        try:
            # Select best model
            model_name = request.model if request.model != "auto" else await self.select_best_model(request.messages)
            
            # Get relevant context from knowledge base
            last_message = request.messages[-1].content
            context_results = await self.search_knowledge_base(last_message, limit=3)
            
            # Enhance messages with context if available
            enhanced_messages = request.messages.copy()
            if context_results:
                context = "\n\n".join([result['content'] for result in context_results[:2]])
                enhanced_messages.insert(0, Message(
                    role="system",
                    content=f"Relevant context for this conversation:\n{context}\n\nUse this context to provide more accurate and helpful responses."
                ))
            
            # Call AI model
            response = await self.call_ai_model(
                model_name,
                enhanced_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Analyze the response
            analysis = await self.analyze_message(response['content'])
            
            # Update model performance
            if model_name not in self.model_performance:
                self.model_performance[model_name] = {
                    'calls': 0,
                    'avg_time': 0,
                    'success_rate': 0
                }
            
            self.model_performance[model_name]['calls'] += 1
            self.model_performance[model_name]['avg_time'] = (
                (self.model_performance[model_name]['avg_time'] * 
                 (self.model_performance[model_name]['calls'] - 1) + 
                 response['processing_time']) / 
                self.model_performance[model_name]['calls']
            )
            
            return ChatResponse(
                response=response['content'],
                model_used=response['model'],
                processing_time=response['processing_time'],
                confidence=analysis.get('sentiment', {}).get('subjectivity', 0.5),
                analysis=analysis
            )
        
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_chat(self, request: ChatRequest):
        """Stream chat response"""
        try:
            model_name = request.model if request.model != "auto" else await self.select_best_model(request.messages)
            
            # For streaming, we'll use a simpler approach
            response = await self.call_ai_model(
                model_name,
                request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Stream the response
            for i, char in enumerate(response['content']):
                yield f"data: {json.dumps({'char': char, 'index': i})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
            yield f"data: {json.dumps({'done': True, 'model': response['model']})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

# Create and run the API server
def create_app():
    api = SHAIAPI()
    return api.app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
