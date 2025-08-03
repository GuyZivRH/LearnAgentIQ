import logging
import os
import tempfile
import subprocess
from typing import Dict, Any, TypedDict

from pydantic import Field
from langgraph.graph import StateGraph, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function
from aiq.profiler.callbacks.langchain_callback_handler import LangchainProfilerHandler

logger = logging.getLogger(__name__)


class CodeGenAgentProperFunctionConfig(FunctionBaseConfig, name="code_gen_agent_proper"):
    """
    AI Code Generation Agent with test-driven development
    """
    reasoning_llm: str = Field(default="reasoning_llm", description="LLM for error analysis and debugging")
    code_llm: str = Field(default="code_llm", description="LLM for code generation")
    max_iterations: int = Field(default=3, description="Maximum number of generation attempts")


class CodeGenState(TypedDict):
    problem_statement: str
    solution_path: str
    test_path: str
    current_code: str
    test_results: str
    error_analysis: str
    iteration_count: int
    max_iterations: int
    success: bool


def create_llm_with_callback(llm_config, callback_name="LLM"):
    """Create LLM instance with LangchainProfilerHandler callback"""
    callback = LangchainProfilerHandler()
    
    # Try different attribute name combinations for NIM config
    api_key = None
    for attr in ['nvidia_api_key', 'api_key']:
        if hasattr(llm_config, attr):
            api_key = getattr(llm_config, attr)
            break
    
    base_url = getattr(llm_config, 'base_url', 
                      getattr(llm_config, 'llm_url', 'https://integrate.api.nvidia.com/v1'))
    
    max_tokens = getattr(llm_config, 'max_tokens', 2048)
    model_name = getattr(llm_config, 'model_name', 'nvidia/llama-3.1-nemotron-70b-instruct')
    
    llm = ChatNVIDIA(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        callbacks=[callback]
    )
    
    logger.info(f"✅ {callback_name} initialized with LangchainProfilerHandler")
    return llm, callback


@register_function(config_type=CodeGenAgentProperFunctionConfig)
async def code_gen_agent_proper_function(
    config: CodeGenAgentProperFunctionConfig, builder: Builder
):
    """
    AI Code Generation Agent that uses LangGraph workflow with test-driven development
    """
    
    # Get LLM configs from builder
    reasoning_llm_config = builder.get_llm_config(config.reasoning_llm)
    code_llm_config = builder.get_llm_config(config.code_llm)
    
    # Create LLM instances with LangchainProfilerHandler callbacks
    reasoning_llm, reasoning_callback = create_llm_with_callback(reasoning_llm_config, "Reasoning LLM")
    code_llm, code_callback = create_llm_with_callback(code_llm_config, "Code LLM")
    
    def read_file_content(file_path: str) -> str:
        """Read content from file, return empty string if file doesn't exist"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ""
    
    def write_file_content(file_path: str, content: str):
        """Write content to file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
    
    def run_tests(solution_path: str, test_path: str) -> str:
        """Run tests against the solution code"""
        try:
            # Create a temporary test runner script
            runner_script = f"""
import sys
import os
sys.path.insert(0, os.path.dirname('{solution_path}'))

# Import the solution
try:
    exec(open('{solution_path}').read(), globals())
except Exception as e:
    print(f"Error importing solution: {{e}}")
    sys.exit(1)

# Run the tests
try:
    exec(open('{test_path}').read(), globals())
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {{e}}")
    sys.exit(1)
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(runner_script)
                temp_script = f.name
            
            result = subprocess.run(
                ['python', temp_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.unlink(temp_script)
            
            if result.returncode == 0:
                return "PASS: " + result.stdout
            else:
                return "FAIL: " + result.stderr + result.stdout
                
        except subprocess.TimeoutExpired:
            return "FAIL: Test execution timeout"
        except Exception as e:
            return f"FAIL: Test execution error: {str(e)}"
    
    @track_function(metadata={"node": "code_generation", "description": "Generate code solution"})
    def generate_code(state: CodeGenState) -> CodeGenState:
        """Generate code solution using the coding LLM"""
        logger.info(f"Generating solution (Attempt {state['iteration_count'] + 1}/{state['max_iterations']})")
        
        existing_code = read_file_content(state['solution_path'])
        test_code = read_file_content(state['test_path'])
        
        prompt = f"""
You are an expert Python programmer. Given the following:

Problem Statement:
{state['problem_statement']}

Current Code (if any):
{existing_code}

Test Code:
{test_code}

{"Previous Error Analysis: " + state['error_analysis'] if state['error_analysis'] else ""}

Generate a complete Python solution that will pass the tests. Return only the Python code, no explanations.
"""
        
        response = code_llm.invoke(prompt)
        generated_code = response.content.strip()
        
        # Log token usage from LangchainProfilerHandler
        logger.info(f"🔢 Code generation tokens: prompt={code_callback.prompt_tokens}, completion={code_callback.completion_tokens}, total={code_callback.total_tokens}")
        
        # Clean up the code (remove markdown formatting if present)
        if generated_code.startswith('```python'):
            generated_code = generated_code[9:]
        if generated_code.endswith('```'):
            generated_code = generated_code[:-3]
        
        generated_code = generated_code.strip()
        
        # Write the generated code to file
        write_file_content(state['solution_path'], generated_code)
        logger.info(f"Generated code: {generated_code[:100]}...")
        
        return {
            **state,
            "current_code": generated_code
        }
    
    @track_function(metadata={"node": "test_execution", "description": "Execute tests against generated code"})
    def execute_tests(state: CodeGenState) -> CodeGenState:
        """Execute tests against the generated code"""
        logger.info("Running tests")
        
        test_results = run_tests(state['solution_path'], state['test_path'])
        logger.info(f"Test results: {test_results[:100]}...")
        
        success = test_results.startswith("PASS")
        
        return {
            **state,
            "test_results": test_results,
            "success": success
        }
    
    @track_function(metadata={"node": "error_analysis", "description": "Analyze errors using reasoning model"})
    def analyze_errors(state: CodeGenState) -> CodeGenState:
        """Analyze test failures using the reasoning model"""
        if state['success']:
            return state
            
        logger.info("Analyzing errors")
        
        prompt = f"""
You are an expert debugging assistant. Analyze the following test failure:

Problem Statement:
{state['problem_statement']}

Generated Code:
{state['current_code']}

Test Results:
{state['test_results']}

Provide a concise analysis of what went wrong and suggestions for fixing the code.
Focus on the specific error and how to resolve it.
"""
        
        response = reasoning_llm.invoke(prompt)
        error_analysis = response.content.strip()
        
        # Log token usage from LangchainProfilerHandler
        logger.info(f"🔢 Error analysis tokens: prompt={reasoning_callback.prompt_tokens}, completion={reasoning_callback.completion_tokens}, total={reasoning_callback.total_tokens}")
        logger.info(f"Error analysis: {error_analysis[:100]}...")
        
        return {
            **state,
            "error_analysis": error_analysis,
            "iteration_count": state['iteration_count'] + 1
        }
    
    def should_continue(state: CodeGenState) -> str:
        """Determine if we should continue iterating"""
        if state['success']:
            return "end"
        elif state['iteration_count'] >= state['max_iterations']:
            return "end"
        else:
            return "generate"
    
    # Build the LangGraph workflow
    workflow = StateGraph(CodeGenState)
    
    # Add nodes
    workflow.add_node("generate", generate_code)
    workflow.add_node("test", execute_tests)
    workflow.add_node("analyze", analyze_errors)
    
    # Add edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "test")
    workflow.add_edge("test", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "generate": "generate",
            "end": END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    @track_function(metadata={"tool": "code_generation", "description": "Main code generation tool"})
    async def _response_fn(input_message: str) -> str:
        """Main function that processes input and runs the code generation workflow"""
        logger.info("Starting code generation task")
        
        # Parse input parameters
        lines = input_message.strip().split('\n')
        params = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                params[key.strip()] = value.strip()
        
        problem_statement = params.get('problem_statement', input_message)
        solution_path = params.get('solution_path', './solution.py')
        test_path = params.get('test_path', './test.py')
        
        # Initialize state
        initial_state = CodeGenState(
            problem_statement=problem_statement,
            solution_path=solution_path,
            test_path=test_path,
            current_code="",
            test_results="",
            error_analysis="",
            iteration_count=0,
            max_iterations=config.max_iterations,
            success=False
        )
        
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Log comprehensive token usage summary from LangchainProfilerHandler
        total_prompt_tokens = code_callback.prompt_tokens + reasoning_callback.prompt_tokens
        total_completion_tokens = code_callback.completion_tokens + reasoning_callback.completion_tokens
        total_all_tokens = code_callback.total_tokens + reasoning_callback.total_tokens
        total_requests = code_callback.successful_requests + reasoning_callback.successful_requests
        
        logger.info(f"🔢 LANGCHAIN PROFILER TOKEN SUMMARY:")
        logger.info(f"🔢   Total Prompt Tokens: {total_prompt_tokens}")
        logger.info(f"🔢   Total Completion Tokens: {total_completion_tokens}")
        logger.info(f"🔢   Total All Tokens: {total_all_tokens}")
        logger.info(f"🔢   Total LLM Requests: {total_requests}")
        logger.info(f"🔢 Code LLM: prompt={code_callback.prompt_tokens}, completion={code_callback.completion_tokens}, total={code_callback.total_tokens}, requests={code_callback.successful_requests}")
        logger.info(f"🔢 Reasoning LLM: prompt={reasoning_callback.prompt_tokens}, completion={reasoning_callback.completion_tokens}, total={reasoning_callback.total_tokens}, requests={reasoning_callback.successful_requests}")
        
        if final_state['success']:
            result = f"✅ Code generation successful!\n\nFinal solution saved to: {solution_path}\n\nTest results: {final_state['test_results']}"
        else:
            result = f"❌ Code generation failed after {config.max_iterations} iterations.\n\nLast attempt saved to: {solution_path}\n\nFinal test results: {final_state['test_results']}"
        
        logger.info("Code generation task completed")
        return result

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        logger.info("Function exited early!")
    finally:
        logger.info("Cleaning up code generation workflow.")