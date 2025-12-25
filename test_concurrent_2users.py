#!/usr/bin/env python3
"""
Test with just 2 concurrent users to see if output is garbled
"""

import asyncio
import aiohttp
import json
import time

URL = "http://localhost:30000/v1/chat/completions"

class UserState:
    def __init__(self, id, prompt):
        self.id = id
        self.prompt = prompt
        self.response = ""
        self.finished = False

    def update(self, delta_text):
        self.response += delta_text

async def test_user(user: UserState):
    """Test a single user"""
    payload = {
        "model": "pearl",
        "messages": [{"role": "user", "content": user.prompt}],
        "max_tokens": 500,
        "temperature": 0,
        "stream": True
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(URL, json=payload) as response:
                if response.status != 200:
                    user.response = f"Error: {response.status}"
                    user.finished = True
                    return
                
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
                            json_str = line[6:]
                            try:
                                data = json.loads(json_str)
                                choices = data.get("choices", [])
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {}).get("content", "")
                                    if delta:
                                        user.update(delta)
                            except (json.JSONDecodeError, IndexError, AttributeError):
                                pass
                        elif line == "data: [DONE]":
                            break
        
        user.finished = True
        
    except Exception as e:
        user.response = f"Error: {str(e)}"
        user.finished = True

async def main():
    # Create 2 users with the same prompt
    users = [
        UserState(0, "Write a haiku about coding."),
        UserState(1, "Write a haiku about coding."),
    ]
    
    print("Starting 2 concurrent requests...")
    print("=" * 80)
    
    # Run concurrently
    await asyncio.gather(*[test_user(u) for u in users])
    
    # Print results
    for user in users:
        print(f"\n{'='*80}")
        print(f"User {user.id + 1}")
        print(f"{'='*80}")
        print(user.response)
        print(f"{'='*80}")
        print(f"Length: {len(user.response)} characters")
        
        # Check for garbled patterns
        if "syllable structure The structure is 5-7-5able" in user.response:
            print("⚠️  GARBLED OUTPUT DETECTED!")
        elif len(user.response) < 100:
            print("⚠️  OUTPUT TOO SHORT!")
        else:
            print("✅ Output looks reasonable")

if __name__ == "__main__":
    asyncio.run(main())
