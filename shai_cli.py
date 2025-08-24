#!/usr/bin/env python3
"""
SHAI (Super Human AI) - Advanced Command Line Interface
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Optional, Dict
from datetime import datetime

# Rich UI
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn

# SHAI Core
from shai_core import SHAICore, ConversationType

console = Console()

class SHAICLI:
    def __init__(self, api_keys: Dict[str, str] = None):
        self.shai = SHAICore(api_keys)
        self.current_conversation_id = None
        self.console = Console()
        
    def display_welcome(self):
        welcome_text = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║                    SHAI - Super Human AI                    ║[/bold cyan]
[bold cyan]║              The Next Generation AI Assistant               ║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]

[bold green]🚀 Advanced Features:[/bold green]
• Multi-model AI intelligence (GPT-4, Claude, Gemini, Cohere)
• Advanced NLP and sentiment analysis
• Knowledge base with RAG capabilities
• Intelligent model selection
• Context-aware responses

[bold yellow]💡 Commands:[/bold yellow]
• Type your message to chat with SHAI
• /help - Show all commands
• /new - Start new conversation
• /list - List conversations
• /save - Save conversation
• /quit - Exit SHAI

[bold blue]🎯 Ready to assist you![/bold blue]
        """
        self.console.print(Panel(welcome_text, title="[bold cyan]Welcome to SHAI[/bold cyan]", border_style="cyan"))
    
    def display_help(self):
        help_text = """
[bold cyan]SHAI Commands[/bold cyan]

/new [type] [title]  - New conversation (types: general, technical, creative, analytical)
/list                - List all conversations
/save [filename]     - Save current conversation
/clear               - Clear current conversation
/help                - Show this help
/quit, /exit         - Exit SHAI

[bold yellow]Examples:[/bold yellow]
/new creative "Story Writing"
/save my_conversation.json
        """
        self.console.print(Panel(help_text, title="[bold cyan]Help[/bold cyan]", border_style="cyan"))
    
    async def process_command(self, command: str) -> bool:
        if not command.startswith('/'):
            return False
        
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            if cmd == '/help':
                self.display_help()
            elif cmd == '/new':
                await self._cmd_new_conversation(args)
            elif cmd == '/list':
                self._cmd_list_conversations()
            elif cmd == '/save':
                self._cmd_save_conversation(args)
            elif cmd == '/clear':
                await self._cmd_clear_conversation()
            elif cmd in ['/quit', '/exit']:
                self.console.print("[yellow]Goodbye! 👋[/yellow]")
                return False
            else:
                self.console.print(f"[red]Unknown command: {cmd}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
        
        return True
    
    async def _cmd_new_conversation(self, args):
        conv_type = ConversationType.GENERAL
        title = "New Conversation"
        
        if args:
            if args[0] in [t.value for t in ConversationType]:
                conv_type = ConversationType(args[0])
                title = " ".join(args[1:]) if len(args) > 1 else f"{conv_type.value.title()} Conversation"
            else:
                title = " ".join(args)
        
        self.current_conversation_id = self.shai.create_conversation(title, conv_type)
        self.console.print(f"[green]Created: {title}[/green]")
    
    def _cmd_list_conversations(self):
        if not self.shai.conversations:
            self.console.print("[yellow]No conversations found.[/yellow]")
            return
        self.shai.display_conversations()
    
    def _cmd_save_conversation(self, args):
        if not self.current_conversation_id:
            self.console.print("[red]No active conversation[/red]")
            return
        
        filename = args[0] if args else f"shai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            conv = self.shai.conversations[self.current_conversation_id]
            data = {
                'id': conv.id,
                'title': conv.title,
                'type': conv.conversation_type.value,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat()
                    }
                    for msg in conv.messages
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.console.print(f"[green]Saved: {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving: {str(e)}[/red]")
    
    async def _cmd_clear_conversation(self):
        if not self.current_conversation_id:
            self.console.print("[red]No active conversation[/red]")
            return
        
        conv = self.shai.conversations[self.current_conversation_id]
        conv.messages.clear()
        conv.updated_at = datetime.now()
        self.console.print("[green]Conversation cleared[/green]")
    
    async def process_message(self, user_input: str):
        if not self.current_conversation_id:
            self.current_conversation_id = self.shai.create_conversation("CLI Conversation")
        
        try:
            with Progress(SpinnerColumn(), console=self.console) as progress:
                task = progress.add_task("SHAI is thinking...", total=None)
                response = await self.shai.process_message(self.current_conversation_id, user_input.strip())
                progress.update(task, completed=True)
            
            self.console.print(f"\n[bold cyan]SHAI:[/bold cyan] {response['response']}")
            self.console.print(f"[dim]Model: {response['model_used']} | Time: {response['processing_time']:.2f}s[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def run(self):
        self.display_welcome()
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                
                if not user_input.strip():
                    continue
                
                # Check if it's a command
                if await self.process_command(user_input):
                    continue
                else:
                    break
                
                # Process as message
                await self.process_message(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /quit to exit[/yellow]")
            except EOFError:
                break

async def main():
    parser = argparse.ArgumentParser(description="SHAI - Super Human AI CLI")
    parser.add_argument("--api-keys", help="Path to API keys JSON file")
    parser.add_argument("--message", "-m", help="Send single message")
    
    args = parser.parse_args()
    
    # Load API keys
    api_keys = {}
    if args.api_keys:
        try:
            with open(args.api_keys, 'r') as f:
                api_keys = json.load(f)
        except Exception as e:
            print(f"Error loading API keys: {e}")
            sys.exit(1)
    else:
        api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'COHERE_API_KEY': os.getenv('COHERE_API_KEY')
        }
    
    cli = SHAICLI(api_keys)
    
    if args.message:
        cli.current_conversation_id = cli.shai.create_conversation("Single Message")
        await cli.process_message(args.message)
    else:
        await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
