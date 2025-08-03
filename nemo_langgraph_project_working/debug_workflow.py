#!/usr/bin/env python3
"""
Clean debugging script for token extraction logic
Works with the existing project structure
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (API key should be set in .env file)
# Ensure NVIDIA_API_KEY is loaded from environment
if 'NVIDIA_API_KEY' not in os.environ:
    raise ValueError("NVIDIA_API_KEY environment variable not set. Please check your .env file.")

def debug_token_extraction_logic():
    """
    Debug the exact token extraction logic from lines 180-182
    """
    print("🔍 DEBUGGING TOKEN EXTRACTION LOGIC")
    print("=" * 50)
    
    # This simulates the exact response.usage_metadata from NVIDIA API
    mock_usage_metadata = {
        'input_tokens': 29, 
        'output_tokens': 682, 
        'total_tokens': 711, 
        'input_token_details': {}, 
        'output_token_details': {}
    }
    
    print(f"📋 Raw usage_metadata from NVIDIA API:")
    print(f"   {mock_usage_metadata}")
    print()
    
    # This is the EXACT code from your workflow lines 180-182
    total_tokens = mock_usage_metadata.get("total_tokens", 0)
    prompt_tokens = mock_usage_metadata.get("input_tokens", 0)  # NVIDIA API uses 'input_tokens'
    completion_tokens = mock_usage_metadata.get("output_tokens", 0)  # NVIDIA API uses 'output_tokens'
    
    print(f"📊 EXTRACTED VALUES:")
    print(f"   total_tokens = {total_tokens}")
    print(f"   prompt_tokens = {prompt_tokens}")
    print(f"   completion_tokens = {completion_tokens}")
    print()
    
    # Verify the extraction is working correctly
    calculated_total = prompt_tokens + completion_tokens
    print(f"🧮 VERIFICATION:")
    print(f"   prompt_tokens + completion_tokens = {calculated_total}")
    print(f"   matches total_tokens? {calculated_total == total_tokens}")
    print()
    
    return total_tokens, prompt_tokens, completion_tokens

def debug_real_llm_call():
    """
    Make a real LLM call to see actual token usage
    """
    print("🔧 DEBUGGING REAL LLM CALL")
    print("=" * 30)
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Create LLM (simplified version without callbacks for debugging)
        llm = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ['NVIDIA_API_KEY'],
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            temperature=0.7
        )
        
        print("📞 Making test LLM call...")
        response = llm.invoke("Briefly analyze: AI safety in autonomous vehicles")
        
        print(f"📋 REAL response.usage_metadata:")
        print(f"   {response.usage_metadata}")
        print()
        
        # Apply the same extraction logic
        if response.usage_metadata:
            total_tokens = response.usage_metadata.get("total_tokens", 0)
            prompt_tokens = response.usage_metadata.get("input_tokens", 0)
            completion_tokens = response.usage_metadata.get("output_tokens", 0)
            
            print(f"📊 REAL TOKEN EXTRACTION:")
            print(f"   total_tokens = {total_tokens}")
            print(f"   prompt_tokens = {prompt_tokens}")
            print(f"   completion_tokens = {completion_tokens}")
            print()
            
            print(f"📄 Response preview:")
            print(f"   {response.content[:200]}...")
        else:
            print("❌ No usage_metadata in response")
            
        return response
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have activated the virtual environment and installed requirements")
        return None
    except Exception as e:
        print(f"❌ LLM call error: {e}")
        return None

def debug_metrics_emission():
    """
    Simulate the metrics emission logic
    """
    print("📊 DEBUGGING METRICS EMISSION")
    print("=" * 35)
    
    # Simulate the metrics emission from your workflow
    mock_usage_metadata = {
        'input_tokens': 29, 
        'output_tokens': 682, 
        'total_tokens': 711
    }
    
    node_name = "analysis_node"
    llm_model = "nvidia/llama-3.1-nemotron-70b-instruct"
    
    # Extract tokens
    total_tokens = mock_usage_metadata.get("total_tokens", 0)
    prompt_tokens = mock_usage_metadata.get("input_tokens", 0)
    completion_tokens = mock_usage_metadata.get("output_tokens", 0)
    
    # Simulate OpenTelemetry labels
    labels = {
        "node_name": node_name,
        "llm_model": llm_model
    }
    
    print(f"🏷️  Metrics labels:")
    print(f"   {labels}")
    print()
    
    print(f"📈 Metrics that would be emitted:")
    print(f"   llm_tokens_total.add({total_tokens}, {labels})")
    print(f"   llm_prompt_tokens_total.add({prompt_tokens}, {labels})")
    print(f"   llm_completion_tokens_total.add({completion_tokens}, {labels})")
    print()

if __name__ == "__main__":
    print("🐛 WORKFLOW DEBUGGING SESSION")
    print("=" * 60)
    print("🎯 Purpose: Debug token extraction logic (lines 180-182)")
    print("📍 Set breakpoints in the functions below for detailed inspection")
    print("=" * 60)
    print()
    
    # Step 1: Debug the extraction logic with mock data
    debug_token_extraction_logic()
    
    # Step 2: Debug with real LLM call
    response = debug_real_llm_call()
    
    # Step 3: Debug metrics emission
    debug_metrics_emission()
    
    print("✅ DEBUGGING SESSION COMPLETE!")
    print()
    print("💡 DEBUGGING TIPS:")
    print("   1. Set breakpoints in any of the functions above")
    print("   2. Use F5 (VSCode) or your IDE's debugger")
    print("   3. Examine variables during execution")
    print("   4. The real LLM call shows actual NVIDIA API response structure") 