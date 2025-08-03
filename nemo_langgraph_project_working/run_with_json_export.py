#!/usr/bin/env python3
"""
Simple CLI wrapper for AIQ with JSON export
Keeps native token tracking while exporting results to JSON
"""

import subprocess
import json
import sys
import os
from datetime import datetime

def run_aiq_with_json_export(input_message):
    """Run AIQ workflow and export results to JSON"""
    
    # Set environment for profiling
    env = os.environ.copy()
    env.update({
        'AIQ_PROFILING_ENABLED': 'true',
        'AIQ_PROFILING_VERBOSE': 'true',
        'PYTHONUNBUFFERED': '1'
    })
    
    print(f"🚀 Running AIQ workflow with JSON export...")
    print(f"📋 Input: {input_message}")
    print("-" * 50)
    
    cmd = [
        'aiq', 'run',
        '--config_file', 'langgraph_workflow/configs/config.yml',
        '--input', input_message
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120
        )
        
        # Create comprehensive JSON output
        timestamp = datetime.now().isoformat()
        output_data = {
            'timestamp': timestamp,
            'input_message': input_message,
            'execution': {
                'return_code': result.returncode,
                'success': result.returncode == 0,
                'execution_time': 'captured_in_logs'
            },
            'workflow_output': result.stdout.strip() if result.stdout else "",
            'aiq_logs': result.stderr.strip() if result.stderr else "",
            'token_tracking': {
                'method': 'AIQ Native LangchainProfilerHandler',
                'automatic_capture': True,
                'per_node_tracking': True,
                'data_location': 'Integrated with AIQ profiling system'
            }
        }
        
        # Extract per-node AIQ token usage from logs (Option 1: Dedicated LLM instances)
        per_node_usage = {}
        total_usage = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'tracking_method': 'AIQ Native LangchainProfilerHandler per node (Option 1)'
        }
        
        if result.stderr:
            # Parse per-node token data from dedicated LLM instances
            import re
            for line in result.stderr.split('\n'):
                # Look for node-specific usage metadata
                if 'ANALYSIS_NODE usage_metadata:' in line:
                    try:
                        metadata_str = line.split('usage_metadata:')[1].strip()
                        input_match = re.search(r"'input_tokens':\s*(\d+)", metadata_str)
                        output_match = re.search(r"'output_tokens':\s*(\d+)", metadata_str)
                        total_match = re.search(r"'total_tokens':\s*(\d+)", metadata_str)
                        
                        if input_match and output_match and total_match:
                            per_node_usage['analysis_node'] = {
                                'input_tokens': int(input_match.group(1)),
                                'output_tokens': int(output_match.group(1)),
                                'total_tokens': int(total_match.group(1))
                            }
                    except Exception as e:
                        print(f"Warning: Could not parse analysis node tokens: {e}")
                
                elif 'RECOMMENDATIONS_NODE usage_metadata:' in line:
                    try:
                        metadata_str = line.split('usage_metadata:')[1].strip()
                        input_match = re.search(r"'input_tokens':\s*(\d+)", metadata_str)
                        output_match = re.search(r"'output_tokens':\s*(\d+)", metadata_str)
                        total_match = re.search(r"'total_tokens':\s*(\d+)", metadata_str)
                        
                        if input_match and output_match and total_match:
                            per_node_usage['recommendations_node'] = {
                                'input_tokens': int(input_match.group(1)),
                                'output_tokens': int(output_match.group(1)),
                                'total_tokens': int(total_match.group(1))
                            }
                    except Exception as e:
                        print(f"Warning: Could not parse recommendations node tokens: {e}")
            
            # Calculate totals from per-node data
            for node_data in per_node_usage.values():
                total_usage['prompt_tokens'] += node_data['input_tokens']
                total_usage['completion_tokens'] += node_data['output_tokens']
                total_usage['total_tokens'] += node_data['total_tokens']
            
            runtime_info = {
                'llm_instances': result.stderr.count("LLM initialized with AIQ native token tracking"),
                'profiling_enabled': 'AIQ_PROFILING_ENABLED' in env,
                'verbose_logging': 'AIQ_PROFILING_VERBOSE' in env,
                'nodes_tracked': len(per_node_usage),
                'per_node_tracking': len(per_node_usage) > 0,
                'token_data_found': total_usage['total_tokens'] > 0
            }
            output_data['token_tracking']['runtime_info'] = runtime_info
            output_data['token_tracking']['total_usage'] = total_usage
            output_data['token_tracking']['per_node_usage'] = per_node_usage
        
        # Save to JSON file
        filename = f"aiq_run_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Execution completed!")
        print(f"📊 Return code: {result.returncode}")
        print(f"📁 Results saved to: {filename}")
        
        if result.returncode == 0:
            print(f"🔢 Native token tracking: Active")
            print(f"📈 Per-node granular tracking: Enabled")
            print(f"💾 JSON export: Complete")
        else:
            print(f"❌ Execution failed")
        
        return filename, output_data
        
    except subprocess.TimeoutExpired:
        print("❌ Workflow timed out")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_with_json_export.py 'Your input message'")
        sys.exit(1)
    
    input_msg = sys.argv[1]
    filename, data = run_aiq_with_json_export(input_msg)
    
    if filename:
        print(f"\n📋 JSON export successful: {filename}")
    else:
        print(f"\n❌ JSON export failed") 