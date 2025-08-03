#!/usr/bin/env python3
"""
Main entry point for the containerized LangGraph AIQ workflow.
Initializes OpenTelemetry and runs the workflow directly.
"""

import json
import sys
import os
from datetime import datetime
import logging
import time

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the refactored workflow function and its config directly
from langgraph_workflow.src.langgraph_workflow.langgraph_workflow_function import get_workflow_response_function, LanggraphWorkflowFunctionConfig

async def main_workflow_run(input_message: str):
    """
    Initializes AIQ workflow and runs it directly.
    """
    logger.info("Starting containerized LangGraph AIQ workflow.")

    # Instantiate the workflow function's config directly
    workflow_config = LanggraphWorkflowFunctionConfig()

    # Call the refactored function to get the response function
    _response_fn = await get_workflow_response_function(config=workflow_config)

    logger.info(f"Running _response_fn with input: {input_message}")
    workflow_result = await _response_fn(input_message)
    logger.info("Workflow execution completed.")
    return workflow_result

if __name__ == "__main__":
    logger.info("Script started in __main__ block.")
    
    if len(sys.argv) < 2:
        input_msg = os.getenv("DEFAULT_INPUT_MESSAGE", "AI safety measures in autonomous vehicles containerized")
        logger.info(f"No input message provided, using default: {input_msg}")
    else:
        input_msg = sys.argv[1]
        logger.info(f"Using provided input message: {input_msg}")

    import asyncio
    try:
        result_output = asyncio.run(main_workflow_run(input_msg))
        print(f"\n--- Workflow Final Output ---\n{result_output}")
        print("\n✅ Workflow completed successfully.")
    except Exception as e:
        logger.error(f"Error running main workflow: {e}", exc_info=True)
        print(f"\n❌ Workflow failed: {e}")

    logger.info("Script finished. Waiting 5 seconds for final OTel metric flush.")
    time.sleep(5) # Allow 5 seconds for final metric flush
    logger.info("Exiting script.")
