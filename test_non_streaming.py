#!/usr/bin/env python3
"""
Compare the exact same prompt with streaming vs non-streaming
"""

import requests
import json

def test_streaming():
    """Test streaming with the same prompt as curl"""
    payload = {
        "prompt": "Explain quantum computing",
        "max_tokens": 256,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "stream": True
    }
    
    print("=" * 80)
    print("STREAMING Response:")
    print("=" * 80)
    
    response = requests.post("http://localhost:30000/v1/completions", json=payload, stream=True)
    
    full_text = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: ") and line != "data: [DONE]":
                json_str = line[6:]
                try:
                    data = json.loads(json_str)
                    choices = data.get("choices", [])
                    if choices and len(choices) > 0:
                        delta = choices[0].get("delta", {}).get("content", "")
                        if delta:
                            full_text += delta
                            print(delta, end="", flush=True)
                except:
                    pass
    
    print("\n" + "=" * 80)
    return full_text

def test_non_streaming():
    """Test non-streaming with the same prompt"""
    payload = {
        "prompt": "Explain quantum computing",
        "max_tokens": 256,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "stream": False
    }
    
    response = requests.post("http://localhost:30000/v1/completions", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print("=" * 80)
        print("NON-STREAMING Response:")
        print("=" * 80)
        print(content)
        print("=" * 80)
        return content
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == "__main__":
    streaming_output = test_streaming()
    print("\n")
    non_streaming_output = test_non_streaming()
    
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print(f"Streaming has <think>: {'<think>' in streaming_output}")
    print(f"Non-streaming has <think>: {'<think>' in non_streaming_output if non_streaming_output else 'N/A'}")
