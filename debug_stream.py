#!/usr/bin/env python3
"""
Debug script to trace streaming output step-by-step for a single user
"""

import asyncio
import aiohttp
import json

URL = "http://localhost:30000/v1/chat/completions"

async def debug_single_stream():
    """Debug a single streaming request"""
    payload = {
        "model": "pearl",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 100,
        "temperature": 0,
        "stream": True
    }
    
    print("=" * 80)
    print("DEBUG: Single User Streaming")
    print("=" * 80)
    
    full_response = ""
    chunk_count = 0
    
    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(URL, json=payload) as response:
                if response.status != 200:
                    print(f"Error: {response.status}")
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
                                        chunk_count += 1
                                        full_response += delta
                                        print(f"\nChunk {chunk_count}:")
                                        print(f"  Delta: {repr(delta)}")
                                        print(f"  Full so far: {repr(full_response[:100])}...")
                            except (json.JSONDecodeError, IndexError, AttributeError) as e:
                                print(f"Error parsing: {e}")
                        elif line == "data: [DONE]":
                            break
        
        print("\n" + "=" * 80)
        print("FINAL OUTPUT:")
        print("=" * 80)
        print(full_response)
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(debug_single_stream())
