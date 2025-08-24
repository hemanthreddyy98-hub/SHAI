#!/usr/bin/env python3
"""
SHAI - Simple Startup Script
Run this to start SHAI immediately!
"""

import os
import sys
import webbrowser
import time
import threading

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    SHAI - Super Human AI                    ║
║              The Next Generation AI Assistant               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting SHAI...")
    
    # Start the simple SHAI server
    try:
        from simple_shai import start_web_server
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            print("🌐 Opening SHAI in your browser...")
            webbrowser.open('http://localhost:8080')
        
        # Start browser thread
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        print("✅ SHAI is ready!")
        print("🌐 SHAI will open in your browser automatically")
        print("📱 You can now chat with SHAI!")
        print("\nPress Ctrl+C to stop SHAI")
        
        # Start the server
        start_web_server()
        
    except ImportError:
        print("❌ Error: Could not import SHAI server")
        print("Make sure you're in the SHAI directory")
    except Exception as e:
        print(f"❌ Error starting SHAI: {e}")
        print("Please check if port 8080 is available")

if __name__ == "__main__":
    main()
