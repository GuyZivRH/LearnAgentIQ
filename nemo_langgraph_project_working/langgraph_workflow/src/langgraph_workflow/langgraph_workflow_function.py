import logging
import json
import os
from typing import Dict, Any
from datetime import datetime
from pydantic import Field, BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.callbacks import CallbackManager
from aiq.profiler.callbacks.langchain_callback_handler import LangchainProfilerHandler

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function

# OpenTelemetry imports for metrics
try:
    from opentelemetry import metrics
    OTEL_METRICS_AVAILABLE = True
except ImportError:
    OTEL_METRICS_AVAILABLE = False
# Optional OpenTelemetry imports - for debugging without Docker
try:
    from opentelemetry.metrics import Counter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ OpenTelemetry not available for debugging: {e}")
    print("🐛 This is OK for debugging - metrics will be logged instead")
    OTEL_AVAILABLE = False
    # Create dummy classes for type hints
    Counter = object
    MeterProvider = object

import os

logger = logging.getLogger(__name__)

# Global OTel instruments (will be populated when _initialize_otel_in_workflow is called)
_otel_meter_instruments: Dict[str, Counter] = {}
_otel_initialized = False

def _initialize_otel_in_workflow():
    """Initializes OpenTelemetry MeterProvider and Counters for the workflow process."""
    global _otel_initialized
    global _otel_meter_instruments

    if _otel_initialized:
        logger.info("OpenTelemetry already initialized in workflow module. Skipping.")
        return

    if not OTEL_AVAILABLE:
        logger.info("🐛 OpenTelemetry not available - debugging mode, metrics will be logged only")
        _otel_initialized = False
        return

    logger.info("Initializing OpenTelemetry within langgraph_workflow_function process.")

    resource = Resource.create({
        "service.name": "langgraph-aiq-workflow-nodes",
        "service.instance.id": os.getenv("HOSTNAME", "localhost"),
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })

    # Get OTLP endpoint from environment variable
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otel_endpoint:
        logger.error("OTEL_EXPORTER_OTLP_ENDPOINT environment variable is not set. OTel will not export!")
        return

    # Configure OTLP metric exporter
    metric_exporter = OTLPMetricExporter(
        endpoint=otel_endpoint,
        insecure=True
    )
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    logger.info(f"OpenTelemetry MeterProvider and MetricReader set in workflow process, sending to: {otel_endpoint}")

    global_meter = metrics.get_meter(__name__)

    _otel_meter_instruments["llm_tokens_total"] = global_meter.create_counter(
        "llm_tokens_total", description="Total tokens consumed by LLM operations (prompt + completion)", unit="token"
    )
    _otel_meter_instruments["llm_prompt_tokens_total"] = global_meter.create_counter(
        "llm_prompt_tokens_total", description="Total prompt tokens consumed by LLM operations", unit="token"
    )
    _otel_meter_instruments["llm_completion_tokens_total"] = global_meter.create_counter(
        "llm_completion_tokens_total", description="Total completion tokens consumed by LLM operations", unit="token"
    )
    logger.info(f"OpenTelemetry metrics instruments created in workflow process: {list(_otel_meter_instruments.keys())}")

    _otel_initialized = True

# State definition for LangGraph
class GraphState(BaseModel):
    topic: str = ""
    analysis: str = ""
    recommendations: str = ""

# Configuration for the LangGraph Workflow Function
class LanggraphWorkflowFunctionConfig(FunctionBaseConfig, name="langgraph_workflow"):
    """
    LangGraph workflow with profiling for AI analysis and recommendations
    """
    # LLM Configuration
    llm_url: str = Field(default="https://integrate.api.nvidia.com/v1", description="LLM API endpoint")
    llm_api_key: str = Field(default=os.getenv("NVIDIA_API_KEY"), description="LLM API key")
    llm_model_name: str = Field(default="nvidia/llama-3.1-nemotron-70b-instruct", description="LLM model name")
    temperature: float = Field(default=0.7, description="LLM temperature")
    
    # Graph prompts
    first_node_prompt: str = Field(default="Analyze the following topic and provide key insights: {topic}", description="Prompt for analysis node")
    second_node_prompt: str = Field(default="Based on the analysis: {analysis}, provide detailed recommendations and action items.", description="Prompt for recommendations node")

# LLM Initialization Function
def create_llm(config: LanggraphWorkflowFunctionConfig):
    aiq_callback = LangchainProfilerHandler()
    
    llm = ChatOpenAI(
        base_url=config.llm_url,
        api_key=config.llm_api_key,
        model=config.llm_model_name,
        temperature=config.temperature,
        callbacks=[aiq_callback]
    )
    
    logger.info(f"🔧 LLM initialized with AIQ native token tracking")
    return llm, aiq_callback

# LangGraph Nodes

@track_function
def analysis_node(state: GraphState, config: LanggraphWorkflowFunctionConfig) -> GraphState:
    """First node: Analyzes the given topic with dedicated AIQ tracking"""
    logger.info("ANALYSIS_NODE: Starting execution.")
    llm, callback = create_llm(config)
    
    prompt = config.first_node_prompt.format(topic=state.topic)
    response = llm.invoke(prompt)
    state.analysis = response.content
    
    logger.info(f"🔍 ANALYSIS_NODE usage_metadata: {response.usage_metadata}")

    # OpenTelemetry Metrics Emission
    if _otel_initialized and _otel_meter_instruments and response.usage_metadata:
        labels = {
            "node_name": "analysis_node",
            "llm_model": config.llm_model_name
        }
        
        total_tokens = response.usage_metadata.get("total_tokens", 0)
        prompt_tokens = response.usage_metadata.get("input_tokens", 0)  # NVIDIA API uses 'input_tokens'
        completion_tokens = response.usage_metadata.get("output_tokens", 0)  # NVIDIA API uses 'output_tokens'

        _otel_meter_instruments["llm_tokens_total"].add(total_tokens, labels)
        _otel_meter_instruments["llm_prompt_tokens_total"].add(prompt_tokens, labels)
        _otel_meter_instruments["llm_completion_tokens_total"].add(completion_tokens, labels)
        
        logger.info(f"📊 Successfully emitted OTel metrics for analysis_node: total={total_tokens}, prompt={prompt_tokens}, completion={completion_tokens}")
    else:
        logger.warning(f"⚠️ OTel metrics NOT emitted for analysis_node. OTel initialized: {_otel_initialized}, instruments populated: {bool(_otel_meter_instruments)}, usage_metadata present: {bool(response.usage_metadata)}")
    
    return state

@track_function
def recommendations_node(state: GraphState, config: LanggraphWorkflowFunctionConfig) -> GraphState:
    """Second node: Generates recommendations with dedicated AIQ tracking"""
    logger.info("RECOMMENDATIONS_NODE: Starting execution.")
    llm, callback = create_llm(config)
    
    prompt = config.second_node_prompt.format(analysis=state.analysis)
    response = llm.invoke(prompt)
    state.recommendations = response.content
    
    logger.info(f"🔍 RECOMMENDATIONS_NODE usage_metadata: {response.usage_metadata}")

    # OpenTelemetry Metrics Emission
    if _otel_initialized and _otel_meter_instruments and response.usage_metadata:
        labels = {
            "node_name": "recommendations_node",
            "llm_model": config.llm_model_name
        }
        
        total_tokens = response.usage_metadata.get("total_tokens", 0)
        prompt_tokens = response.usage_metadata.get("input_tokens", 0)  # NVIDIA API uses 'input_tokens'
        completion_tokens = response.usage_metadata.get("output_tokens", 0)  # NVIDIA API uses 'output_tokens'

        _otel_meter_instruments["llm_tokens_total"].add(total_tokens, labels)
        _otel_meter_instruments["llm_prompt_tokens_total"].add(prompt_tokens, labels)
        _otel_meter_instruments["llm_completion_tokens_total"].add(completion_tokens, labels)

        logger.info(f"📊 Successfully emitted OTel metrics for recommendations_node: total={total_tokens}, prompt={prompt_tokens}, completion={completion_tokens}")
    else:
        logger.warning(f"⚠️ OTel metrics NOT emitted for recommendations_node. OTel initialized: {_otel_initialized}, instruments populated: {bool(_otel_meter_instruments)}, usage_metadata present: {bool(response.usage_metadata)}")
    
    return state

# LangGraph Graph Creation
def create_graph(config: LanggraphWorkflowFunctionConfig):
    workflow = StateGraph(GraphState)
    workflow.add_node("analyze", lambda state: analysis_node(state, config))
    workflow.add_node("recommend", lambda state: recommendations_node(state, config))
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", END)
    return workflow.compile()

# MODIFIED: Direct Workflow Function for telemetry script
async def get_workflow_response_function(config: LanggraphWorkflowFunctionConfig):
    """
    This function now directly returns the async response function for the LangGraph workflow.
    It is used by the telemetry script (run_with_json_export_tel.py).
    """
    logger.info("get_workflow_response_function: Initializing OTel and preparing response function.")
    _initialize_otel_in_workflow()

    @track_function
    async def _response_fn(input_message: str) -> str:
        logger.info(f"_response_fn: Starting for input: {input_message}")
        graph = create_graph(config)
        initial_state = GraphState(topic=input_message)
        result = graph.invoke(initial_state)
        
        logger.info(f"📊 PER_NODE_TRACKING: Graph execution completed with dedicated LLM instances per node")
        
        output = f"""🚀 LangGraph Analysis Results for: {input_message}
{"=" * 60}

📊 PER-NODE TOKEN TRACKING (AIQ Native & OpenTelemetry):
{"-" * 40}
✅ Each node uses dedicated LLM instance with AIQ tracking
📋 Token usage captured per node in logs (AIQ)
🔍 Detailed per-node metrics available via OpenTelemetry for Prometheus/Grafana
💡 Uses pure AIQ Native LangchainProfilerHandler per node for AIQ logs

📊 ANALYSIS:
{"-" * 40}
{result['analysis']}

💡 RECOMMENDATIONS:
{"-" * 40}
{result['recommendations']}

✅ Graph execution completed with per-node AIQ token tracking and OTel metrics!"""
        
        logger.info("_response_fn: Workflow execution complete.")
        return output

    return _response_fn

# Original AIQ Registration (for aiq run compatibility)
@register_function(config_type=LanggraphWorkflowFunctionConfig)
async def langgraph_workflow_function(
    config: LanggraphWorkflowFunctionConfig, builder: Builder
):
    """AIQ Workflow with comprehensive token tracking"""
    
    # Use the same function as the telemetry script
    _response_fn = await get_workflow_response_function(config)
    
    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        logger.info("Token-tracked workflow exited early!")
    finally:
        logger.info("Cleaning up token-tracked workflow.")


# DEBUG MAIN BLOCK - For direct debugging of this file
if __name__ == "__main__":
    import asyncio
    import os
    
    print("🐛 DIRECT DEBUG MODE - langgraph_workflow_function.py")
    print("=" * 60)
    print("🎯 Set breakpoints in analysis_node() or recommendations_node()")
    print("📍 Line 173 is in recommendations_node() - token extraction logic")
    print("=" * 60)
    
    # Set environment variables for debugging
    # NVIDIA_API_KEY should be loaded from .env file
    if 'NVIDIA_API_KEY' not in os.environ:
        raise ValueError("NVIDIA_API_KEY environment variable not set. Please check your .env file.")
    os.environ['ENVIRONMENT'] = 'development'
    
    async def debug_main():
        print("🚀 Starting direct workflow execution...")
        
        # Create config
        config = LanggraphWorkflowFunctionConfig()
        
        # Get the workflow function
        print("🔧 Getting workflow response function...")
        _response_fn = await get_workflow_response_function(config)
        
        # Execute with debug input
        test_input = "AI safety debugging - direct execution"
        print(f"📋 Input: {test_input}")
        print("🎯 Execution will hit your breakpoints in analysis_node and recommendations_node!")
        print("-" * 40)
        
        # THIS WILL HIT YOUR BREAKPOINTS!
        result = await _response_fn(test_input)
        
        print("\n✅ Debug execution completed!")
        print(f"📄 Result: {result}")
        
        return result
    
    # Run the debug workflow
    try:
        asyncio.run(debug_main())
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted by user")
    except Exception as e:
        print(f"\n❌ Debug error: {e}")
        import traceback
        traceback.print_exc() 