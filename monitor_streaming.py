#!/usr/bin/env python3
"""
Streaming Monitor for nano-PEARL Server
Simulates multiple concurrent users and displays their streaming responses in a TUI.
"""

import asyncio
import aiohttp
import json
import argparse
import random
import time
from datetime import datetime
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from rich.columns import Columns
from rich import box

# Defaults (can be overridden via CLI)
URL = "http://localhost:30000/v1/chat/completions"
NUM_USERS = 4
MODEL_NAME = "pearl"
MAX_TOKENS = 500

# Prompts to simulate user activity
PROMPTS = [
    "Write a haiku about a coding bug.",
    "What are the benefits of distributed systems?",
    "Describe a futuristic city with flying cars.",
    "How does a neural network learn?",
    "Write a short story about a robot who loves gardening.",
    "Explain the difference between TCP and UDP.",
    "What is the meaning of life, the universe, and everything?",
    "Write a python function to calculate Fibonacci numbers.",
    "Describe the taste of a fresh strawberry."
]

class UserState:
    def __init__(self, id, prompt):
        self.id = id
        self.prompt = prompt
        self.response = ""
        self.status = "Waiting..."
        self.start_time = None
        self.tokens = 0
        self.finished = False

    def update(self, delta_text):
        self.response += delta_text
        self.tokens += 1

    def finish(self):
        self.finished = True
        self.status = "Completed"

def make_layout(users):
    """Render a responsive grid of user panels (scales to many users).
    - <=4 users: show streaming content.
    - >4 users: hide content body, only show status and token count.
    """
    show_full = len(users) <= 4
    panels = []
    for user in users:
        border_style = "blue"
        if user.status == "Streaming":
            border_style = "green"
        elif user.status == "Completed":
            border_style = "bright_black"
        elif user.status.startswith("Error"):
            border_style = "red"

        content = Text()
        trimmed_prompt = user.prompt if len(user.prompt) <= 64 else f"{user.prompt[:64]}..."
        content.append(f"Prompt: {trimmed_prompt}\n", style="bold yellow")
        content.append(f"Status: {user.status} | Chunks: {user.tokens}\n", style="italic cyan")
        if show_full:
            content.append("-" * 40 + "\n", style="dim")
            content.append(user.response)

        panels.append(
            Panel(
                content,
                title=f"User {user.id + 1}",
                border_style=border_style,
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    # Columns will wrap panels to fit terminal width; good for large N.
    return Columns(panels, expand=True, equal=True)

async def simulate_user(user: UserState):
    """Simulates a single user sending a streaming request."""
    user.status = "Connecting..."
    user.start_time = time.time()
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": user.prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(URL, json=payload) as response:
                if response.status != 200:
                    user.status = f"Error: {response.status}"
                    user.response = await response.text()
                    return

                user.status = "Streaming"
                
                # Buffer to accumulate incomplete lines
                buffer = b""
                
                async for chunk in response.content:
                    # Add chunk to buffer
                    buffer += chunk
                    
                    # Process all complete lines in buffer
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith("data: ") and line != "data: [DONE]":
                            json_str = line[6:]  # Skip "data: "
                            try:
                                data = json.loads(json_str)
                                choices = data.get("choices", [])
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {}).get("content", "")
                                    if delta:
                                        user.update(delta)
                            except (json.JSONDecodeError, IndexError, AttributeError) as e:
                                # Silently skip malformed chunks
                                pass
                        elif line == "data: [DONE]":
                            break
                        
        user.finish()
        
    except Exception as e:
        user.status = f"Error: {str(e)}"
        user.finished = True

async def monitor_loop():
    """Main loop handling the UI and user tasks."""
    console = Console()
    users = [UserState(i, random.choice(PROMPTS)) for i in range(NUM_USERS)]

    with Live(make_layout(users), refresh_per_second=10, screen=True) as live:
        tasks = [asyncio.create_task(simulate_user(u)) for u in users]

        while not all(u.finished for u in users):
            live.update(make_layout(users))
            await asyncio.sleep(0.1)

        live.update(make_layout(users))
        await asyncio.gather(*tasks)

    console.clear()
    console.print(Panel(Text("All tasks completed. Full results below:", style="bold green"), box=box.HEAVY))

    for user in users:
        title_style = "red" if user.status.startswith("Error") else "green"
        console.print(f"\n[bold {title_style}]User {user.id + 1} ({user.status})[/bold {title_style}]")
        console.print(f"[bold yellow]Prompt:[/bold yellow] {user.prompt}")
        console.print(f"[bold cyan]Token Count:[/bold cyan] {user.tokens}")
        console.print("-" * 80)
        console.print(user.response)
        console.print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Monitor for nano-PEARL Server")
    parser.add_argument("--url", default=URL, help="Completions endpoint")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="max_tokens per request")
    parser.add_argument("--users", type=int, default=NUM_USERS, help="Number of concurrent users to simulate")
    args = parser.parse_args()

    URL = args.url
    MODEL_NAME = args.model
    MAX_TOKENS = args.max_tokens
    NUM_USERS = max(1, args.users)

    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        print("\nStopped by user.")
