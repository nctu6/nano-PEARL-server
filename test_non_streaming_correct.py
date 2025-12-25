#!/usr/bin/env python3
"""
Test non-streaming mode to check if output is correct
"""

import requests
import json

def test_non_streaming_chat():
    """Test non-streaming with chat endpoint"""
    payload = {
        "model": "pearl",
        "messages": [{"role": "user", "content": "Write a haiku about coding."}],
        "max_tokens": 200,
        "temperature": 0,
        "stream": False
    }
    
    print("=" * 80)
    print("Non-Streaming Chat Completions Test")
    print("=" * 80)
    
    response = requests.post("http://localhost:30000/v1/chat/completions", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(content)
        print("=" * 80)
        print(f"Length: {len(content)} characters")
        print("=" * 80)
        
        # Check for garbled text patterns
        if "able structure is 5-7-5" in content or "syllables. Maybe" in content:
            print("⚠️  WARNING: Output appears garbled!")
        else:
            print("✅ Output looks clean!")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_non_streaming_completions():
    """Test non-streaming with completions endpoint"""
    payload = {
        "prompt": "Write a haiku about coding.",
        "max_tokens": 200,
        "temperature": 0,
        "stream": False
    }
    
    print("\n" + "=" * 80)
    print("Non-Streaming Completions Test")
    print("=" * 80)
    
    response = requests.post("http://localhost:30000/v1/completions", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(content)
        print("=" * 80)
        print(f"Length: {len(content)} characters")
        print("=" * 80)
        
        # Check for garbled text patterns
        if "able structure is 5-7-5" in content or "syllables. Maybe" in content:
            print("⚠️  WARNING: Output appears garbled!")
        else:
            print("✅ Output looks clean!")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_non_streaming_chat()
    test_non_streaming_completions()
