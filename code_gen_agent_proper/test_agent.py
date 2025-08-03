#!/usr/bin/env python3
"""
Test script for the AI Code Generation Agent
"""

import asyncio
import os
from pathlib import Path

# Add the source directory to the path
import sys
sys.path.insert(0, 'src')

async def test_code_generation():
    """Test the code generation agent directly"""
    
    print("🧪 Testing AI Code Generation Agent")
    print("=" * 50)
    
    # Get absolute paths
    current_dir = Path(__file__).parent
    test_path = str(current_dir / "test_examples" / "rectangle_tests.py")
    solution_path = str(current_dir / "test_examples" / "rectangle_solution.py")
    
    problem_statement = """
Write a Python function named largest_rectangle that computes the area of the largest rectangle in a histogram.
Given an array heights of non-negative integers representing the histogram bar heights where the width of each bar is 1,
return the area of the largest rectangle that can be formed within the histogram.

Special requirements:
- Return -1 for empty arrays
- Use an efficient algorithm (stack-based approach recommended)
"""
    
    input_message = f"""problem_statement: {problem_statement.strip()}
solution_path: {solution_path}
test_path: {test_path}"""
    
    print(f"📝 Problem: Largest Rectangle in Histogram")
    print(f"🧪 Test file: {test_path}")
    print(f"💾 Solution file: {solution_path}")
    print()
    
    print("📋 Current test file content:")
    try:
        with open(test_path, 'r') as f:
            content = f.read()
            print(content[:500] + "..." if len(content) > 500 else content)
    except FileNotFoundError:
        print("Test file not found!")
    
    print("\n💡 Current solution (before AI generation):")
    try:
        with open(solution_path, 'r') as f:
            content = f.read()
            print(content)
    except FileNotFoundError:
        print("Solution file not found!")
    
    print("\n🤖 This demonstrates the setup for our AI Code Generation Agent.")
    print("Input format that would be sent to the agent:")
    print("-" * 30)
    print(input_message)
    print("-" * 30)
    
    print("\n🎯 The agent would:")
    print("1. Parse the problem statement, solution path, and test path")
    print("2. Generate code using the coding LLM (Qwen2.5-Coder)")
    print("3. Run tests against the generated code")
    print("4. If tests fail, analyze errors with reasoning LLM (DeepSeek-R1)")
    print("5. Iterate until success or max attempts reached")
    
    print(f"\n✅ Test setup complete! The agent is ready to generate code.")

if __name__ == "__main__":
    asyncio.run(test_code_generation()) 