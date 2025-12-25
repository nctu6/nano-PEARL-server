#!/usr/bin/env python3
"""
Test single user to see if output is clean
"""

import asyncio
import aiohttp
import json

URL = "http://localhost:30000/v1/chat/completions"

async def test_single_user():
    """Test a single user request"""
    payload = {
        "model": "pearl",
        "messages": [{"role": "user", "content": "Write a haiku about coding."}],
        "max_tokens": 200,
        "temperature": 0,
        "stream": True
    }
    
    print("=" * 80)
    print("Single User Streaming Test")
    print("=" * 80)
    
    full_response = ""
    
    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(URL, json=payload) as response:
                if response.status != 200:
                    print(f"Error: {response.status}")
                    print(await response.text())
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
                                        full_response += delta
                                        print(delta, end="", flush=True)
                            except (json.JSONDecodeError, IndexError, AttributeError):
                                pass
                        elif line == "data: [DONE]":
                            break
        
        print("\n" + "=" * 80)
        print(f"Total characters: {len(full_response)}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_single_user())
