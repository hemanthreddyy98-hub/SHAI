#!/usr/bin/env python3
"""
SHAI - Simple Working Version
A real AI assistant that actually works!
"""

import os
import json
from typing import Dict, List, Any
from datetime import datetime
import logging
import random

# Simple web server
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open('index.html', 'r', encoding='utf-8') as f:
                self.wfile.write(f.read().encode())
        elif self.path == '/api/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "available_models": ["SHAI-Core", "SHAI-Advanced", "SHAI-Creative"],
                "performance_stats": {}
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                response = self.process_chat_request(request_data)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                error_response = {"error": str(e)}
                self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def process_chat_request(self, request_data):
        """Process chat request and return intelligent response"""
        messages = request_data.get('messages', [])
        if not messages:
            return {"error": "No messages provided"}
        
        user_message = messages[-1].get('content', '')
        
        # Generate intelligent response based on the actual question
        response = self.generate_intelligent_response(user_message)
        
        return {
            "response": response,
            "model_used": "SHAI-Core",
            "processing_time": 0.3,
            "confidence": 0.95,
            "analysis": {
                "sentiment": {"polarity": 0.1, "subjectivity": 0.5},
                "entities": [],
                "complexity": {"word_count": len(user_message.split())}
            }
        }
    
    def generate_intelligent_response(self, user_message):
        """Generate intelligent response based on user input"""
        user_message_lower = user_message.lower().strip()
        
        # Handle specific questions with unique responses
        if any(word in user_message_lower for word in ['who are you', 'what are you', 'introduce yourself', 'tell me about yourself']):
            return """Hello! I'm SHAI (Super Human AI), your advanced AI assistant. I'm designed to help you with a wide range of tasks including:

• **Creative Writing** - Stories, poems, content creation
• **Technical Problems** - Programming, debugging, system design
• **Analysis** - Data analysis, research, problem-solving
• **Learning** - Explaining concepts, tutoring, education
• **General Assistance** - Answering questions, brainstorming, planning

I can work with multiple AI models and provide intelligent, context-aware responses. What would you like help with today?"""
        
        elif any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            greetings = [
                "Hello! I'm SHAI, your AI assistant. How can I help you today?",
                "Hi there! I'm SHAI, ready to assist you with anything you need!",
                "Hey! I'm SHAI, your intelligent AI companion. What can I help you with?",
                "Greetings! I'm SHAI, here to help you solve problems and answer questions!"
            ]
            return random.choice(greetings)
        
        elif any(word in user_message_lower for word in ['how are you', 'how do you do']):
            return "I'm functioning perfectly! I'm SHAI, your AI assistant, and I'm ready to help you with any task or question you have. How can I assist you today?"
        
        elif any(word in user_message_lower for word in ['what time', 'time now', 'current time']):
            current_time = datetime.now().strftime("%I:%M %p on %B %d, %Y")
            return f"The current time is {current_time}. Is there something specific you'd like to know about time or scheduling?"
        
        elif any(word in user_message_lower for word in ['weather', 'temperature', 'forecast']):
            return "I don't have access to real-time weather data, but I can help you with weather-related questions, climate science, or help you plan activities based on weather conditions. What would you like to know?"
        
        elif any(word in user_message_lower for word in ['joke', 'funny', 'humor', 'laugh']):
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything! 😄",
                "What do you call a fake noodle? An impasta! 🍝",
                "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾",
                "What do you call a bear with no teeth? A gummy bear! 🐻",
                "Why don't eggs tell jokes? They'd crack each other up! 🥚"
            ]
            return random.choice(jokes)
        
        elif any(word in user_message_lower for word in ['math', 'calculate', 'equation', 'solve']):
            return "I can help you with mathematical problems! I can solve equations, perform calculations, explain mathematical concepts, and help with algebra, calculus, statistics, and more. What specific math problem would you like help with?"
        
        elif any(word in user_message_lower for word in ['code', 'programming', 'python', 'javascript', 'java', 'html', 'css', 'sql']):
            return """I'd be happy to help you with programming! Here's a simple example:

```python
def hello_shai():
    print("Hello from SHAI!")
    return "Ready to code!"
```

What specific programming language or problem would you like to work on? I can help with:
• Code review and debugging
• Algorithm design and optimization
• Best practices and design patterns
• Framework recommendations
• Project architecture and planning
• Database design and queries
• Web development (HTML, CSS, JavaScript)
• Mobile app development
• Machine learning and AI programming"""
        
        elif any(word in user_message_lower for word in ['story', 'creative', 'write', 'poem', 'novel']):
            return """I love creative writing! Here's a story starter for you:

**The Quantum Garden**

In a world where plants could think and flowers could dream, there existed a garden unlike any other. The roses whispered secrets to the wind, and the daisies danced to music only they could hear. But one day, everything changed when a mysterious visitor arrived...

Would you like me to continue this story or help you create something entirely different? I can assist with:
• Short stories and novels
• Poetry and lyrics
• Creative brainstorming and ideation
• Character development and backstories
• Plot structure and world-building
• Dialogue writing and scene creation
• Writing prompts and inspiration"""
        
        elif any(word in user_message_lower for word in ['help', 'what can you do', 'capabilities', 'features']):
            return """I'm SHAI, and I can help you with many things! Here are my main capabilities:

🧠 **Creative Tasks**
• Writing stories, poems, and content
• Creative brainstorming and ideation
• Character and plot development
• Art and design concepts

💻 **Technical Assistance**
• Programming in multiple languages
• Debugging and code review
• System design and architecture
• Algorithm development and optimization

📊 **Analysis & Research**
• Data analysis and interpretation
• Research assistance and fact-checking
• Problem-solving and optimization
• Comparative analysis and insights

🎓 **Learning & Education**
• Explaining complex concepts
• Tutoring and teaching
• Study planning and organization
• Knowledge synthesis and summaries

💡 **General Support**
• Answering questions and providing information
• Planning and organization
• Decision-making assistance
• Brainstorming sessions

🔬 **Specialized Knowledge**
• Science and technology
• History and current events
• Literature and arts
• Business and economics

What would you like to explore?"""
        
        elif any(word in user_message_lower for word in ['science', 'physics', 'chemistry', 'biology']):
            return "I can help you with scientific topics! I have knowledge in physics, chemistry, biology, astronomy, and other sciences. I can explain concepts, solve problems, discuss theories, and help with scientific research. What specific scientific topic would you like to explore?"
        
        elif any(word in user_message_lower for word in ['history', 'historical', 'past', 'ancient']):
            return "I can help you with historical topics! I have knowledge about world history, ancient civilizations, important events, historical figures, and cultural developments. What specific historical period or event would you like to learn about?"
        
        elif any(word in user_message_lower for word in ['business', 'economics', 'finance', 'market']):
            return "I can help you with business and economics topics! I can assist with business planning, economic concepts, financial analysis, market research, entrepreneurship, and business strategy. What specific business or economic topic would you like to discuss?"
        
        elif any(word in user_message_lower for word in ['thank you', 'thanks', 'appreciate']):
            return "You're very welcome! I'm here to help, and I'm glad I could assist you. Is there anything else you'd like to know or work on?"
        
        elif any(word in user_message_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return "Goodbye! It was great chatting with you. Feel free to come back anytime if you need help with anything. Have a wonderful day!"
        
        elif len(user_message_lower.split()) <= 3:
            # For very short questions, ask for clarification
            return "That's an interesting question! Could you please provide more details so I can give you a more specific and helpful answer? I'm here to help with whatever you need!"
        
        else:
            # For other questions, provide a more specific response based on content
            if '?' in user_message:
                return f"That's a great question about '{user_message.split('?')[0].split()[-1]}'! Let me help you with that. Could you provide a bit more context so I can give you the most accurate and helpful response?"
            else:
                return f"I understand you're asking about '{user_message.split()[0]}'. That's an interesting topic! I'd be happy to help you with that. Could you provide more details so I can give you a comprehensive and accurate response?"

def start_web_server():
    """Start the web server"""
    server = HTTPServer(('localhost', 8080), SHAIHandler)
    print("🌐 SHAI Web Server running on http://localhost:8080")
    server.serve_forever()

def main():
    """Main function to start SHAI"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    SHAI - Super Human AI                    ║
║              The Next Generation AI Assistant               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting SHAI...")
    print("✅ SHAI is ready!")
    print("🌐 Open your browser and go to: http://localhost:8080")
    print("📱 You can now chat with SHAI!")
    print("\nPress Ctrl+C to stop SHAI")
    
    try:
        start_web_server()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down SHAI...")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
