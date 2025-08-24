#!/usr/bin/env python3
"""
SHAI Startup Script - Launch the complete SHAI system
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'openai', 'google-generativeai', 
        'anthropic', 'cohere', 'transformers', 'torch', 'spacy',
        'sentence-transformers', 'textblob', 'nltk', 'chromadb',
        'rich', 'structlog'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies are installed!")

def check_api_keys():
    """Check if API keys are set"""
    required_keys = [
        'OPENAI_API_KEY',
        'GEMINI_API_KEY', 
        'ANTHROPIC_API_KEY',
        'COHERE_API_KEY'
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("SHAI will work with available models only.")
        print("To get API keys:")
        print("  - OpenAI: https://platform.openai.com/api-keys")
        print("  - Gemini: https://makersuite.google.com/app/apikey")
        print("  - Anthropic: https://console.anthropic.com/")
        print("  - Cohere: https://dashboard.cohere.ai/api-keys")
    else:
        print("✅ All API keys are configured!")

def start_api_server():
    """Start the SHAI API server"""
    print("🚀 Starting SHAI API server...")
    try:
        subprocess.run([
            sys.executable, 'shai_api.py'
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def start_web_server():
    """Start the web interface server"""
    print("🌐 Starting web interface...")
    try:
        subprocess.run([
            sys.executable, '-m', 'http.server', '8080'
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Web server stopped")
    except Exception as e:
        print(f"❌ Error starting web server: {e}")

def open_browser():
    """Open browser to SHAI interface"""
    time.sleep(3)  # Wait for servers to start
    print("🌐 Opening SHAI in browser...")
    webbrowser.open('http://localhost:8080')

def main():
    """Main startup function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    SHAI - Super Human AI                    ║
║              The Next Generation AI Assistant               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    check_dependencies()
    
    # Check API keys
    print("\n🔑 Checking API keys...")
    check_api_keys()
    
    # Start API server in background
    print("\n🚀 Starting SHAI system...")
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Start web server in background
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Open browser
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print("""
✅ SHAI is starting up!

📡 API Server: http://localhost:8000
🌐 Web Interface: http://localhost:8080
📚 API Documentation: http://localhost:8000/docs

Press Ctrl+C to stop SHAI
    """)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down SHAI...")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
