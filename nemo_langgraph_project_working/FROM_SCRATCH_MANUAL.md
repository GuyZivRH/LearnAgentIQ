# 🏗️ Build NeMo LangGraph with Monitoring from Scratch

A **complete step-by-step guide** to build a production-ready LangGraph workflow with AIQ integration, OpenTelemetry metrics, Prometheus monitoring, and Grafana visualization.

## 📋 Table of Contents

1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Project Structure Creation](#2-project-structure-creation)
3. [Python Dependencies](#3-python-dependencies)
4. [Core LangGraph Implementation](#4-core-langgraph-implementation)
5. [AIQ Workflow Registration](#5-aiq-workflow-registration)
6. [OpenTelemetry Integration](#6-opentelemetry-integration)
7. [Docker Configuration](#7-docker-configuration)
8. [Monitoring Stack Setup](#8-monitoring-stack-setup)
9. [Testing & Validation](#9-testing--validation)
10. [Production Deployment](#10-production-deployment)

---

## 1. Prerequisites & Setup

### **1.1 System Requirements**
```bash
# Required software
- Docker Desktop (latest version)
- Docker Compose (v2.0+)
- Python 3.11+ (for local development)
- Git (for version control)
- Text editor (VSCode recommended)
```

### **1.2 API Keys & Access**
```bash
# NVIDIA API access
NVIDIA_API_KEY=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
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
LLM_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
```

### **1.3 Why Docker for This Project?**
```text
🐳 DOCKER BENEFITS:
==================

1. ISOLATION:
   - Separate environments for each service
   - No dependency conflicts between components
   - Clean separation of concerns

2. SCALABILITY:
   - Easy horizontal scaling of services
   - Independent resource allocation
   - Container orchestration ready

3. REPRODUCIBILITY:
   - Identical environments across dev/prod
   - Version-controlled infrastructure
   - Consistent deployments

4. MONITORING STACK:
   - Pre-configured Prometheus + Grafana
   - Service discovery via Docker networks
   - Automatic container health checks

5. MAINTENANCE:
   - Easy updates and rollbacks
   - Simplified dependency management
   - Portable across platforms
```

---

## 2. Project Structure Creation

### **2.1 Create Root Directory**
```bash
# Create main project directory
mkdir nemo_langgraph_project
cd nemo_langgraph_project

# Create basic structure
mkdir -p langgraph_workflow/configs
mkdir -p langgraph_workflow/src/langgraph_workflow
mkdir venv
```

### **2.2 Initialize Git Repository**
```bash
git init
echo "venv/" > .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore
echo "*.log" >> .gitignore
```

### **2.3 Final Directory Structure**
```
nemo_langgraph_project/
├── 📄 requirements.txt                   # Python dependencies
├── 🐳 Dockerfile                         # App container definition
├── 🐳 docker-compose.yml                # Multi-service orchestration
├── ⚙️ .env                               # Environment variables
├── ⚙️ prometheus.yml                     # Prometheus configuration
├── ⚙️ otel-collector-config.yaml        # OpenTelemetry configuration
├── 🐍 run_with_json_export_tel.py       # Main execution script
├── 📁 venv/                             # Virtual environment
└── 📁 langgraph_workflow/               # AIQ Workflow Package
    ├── 📄 pyproject.toml                # Package definition
    ├── 📁 configs/
    │   └── 📄 config.yml                # AIQ configuration
    └── 📁 src/langgraph_workflow/
        ├── 📄 __init__.py               # Package marker
        ├── 📄 register.py               # Component registration
        └── 📄 langgraph_workflow_function.py  # Main workflow
```

---

## 3. Python Dependencies

### **3.1 Create Requirements File**
```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core AI/ML frameworks
aiqtoolkit
langgraph
langchain
langchain-openai
langchain-community
langchain-core

# Utility libraries
requests
pydantic
python-dotenv
pyyaml

# OpenTelemetry stack
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-exporter-otlp-proto-grpc
opentelemetry-instrumentation-langchain
EOF
```

### **3.2 Create Virtual Environment (Optional)**
```bash
# Create virtual environment for local development
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Core LangGraph Implementation

### **4.1 Create Package Structure**
```bash
# Create package files
touch langgraph_workflow/src/langgraph_workflow/__init__.py
```

### **4.2 Define State Model**
```python
# File: langgraph_workflow/src/langgraph_workflow/langgraph_workflow_function.py

import logging
from typing import Dict, Any
from pydantic import Field, BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.callbacks import CallbackManager
import os

# AIQ Toolkit imports
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function
from aiq.profiler.callbacks.langchain_callback_handler import LangchainProfilerHandler

logger = logging.getLogger(__name__)

# State definition for LangGraph
class GraphState(BaseModel):
    topic: str = ""
    analysis: str = ""
    recommendations: str = ""
```

### **4.3 Add Configuration Class**
```python
# Add to langgraph_workflow_function.py

class LanggraphWorkflowFunctionConfig(FunctionBaseConfig, name="langgraph_workflow"):
    """
    Configuration for LangGraph workflow with AI analysis and recommendations
    """
    # LLM Configuration
    llm_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", 
        description="LLM API endpoint"
    )
    llm_api_key: str = Field(
        default=os.getenv("NVIDIA_API_KEY", "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"), 
        description="LLM API key"
    )
    llm_model_name: str = Field(
        default="nvidia/llama-3.1-nemotron-70b-instruct", 
        description="LLM model name"
    )
    temperature: float = Field(
        default=0.7, 
        description="LLM temperature"
    )
    
    # Node prompts
    first_node_prompt: str = Field(
        default="Analyze the following topic and provide key insights: {topic}", 
        description="Prompt for analysis node"
    )
    second_node_prompt: str = Field(
        default="Based on the analysis: {analysis}, provide detailed recommendations and action items.", 
        description="Prompt for recommendations node"
    )
```

### **4.4 Create LLM Factory Function**
```python
# Add to langgraph_workflow_function.py

def create_llm(config: LanggraphWorkflowFunctionConfig):
    """Create LLM instance with AIQ callback for token tracking"""
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
```

### **4.5 Implement Graph Nodes**
```python
# Add to langgraph_workflow_function.py

@track_function
def analysis_node(state: GraphState, config: LanggraphWorkflowFunctionConfig) -> GraphState:
    """First node: Analyzes the given topic"""
    logger.info("ANALYSIS_NODE: Starting execution.")
    
    # Create dedicated LLM instance for this node
    llm, callback = create_llm(config)
    
    prompt = config.first_node_prompt.format(topic=state.topic)
    response = llm.invoke(prompt)
    state.analysis = response.content
    
    # Log token usage for monitoring
    logger.info(f"🔍 ANALYSIS_NODE usage_metadata: {response.usage_metadata}")
    
    return state

@track_function  
def recommendations_node(state: GraphState, config: LanggraphWorkflowFunctionConfig) -> GraphState:
    """Second node: Generates recommendations based on analysis"""
    logger.info("RECOMMENDATIONS_NODE: Starting execution.")
    
    # Create dedicated LLM instance for this node
    llm, callback = create_llm(config)
    
    prompt = config.second_node_prompt.format(analysis=state.analysis)
    response = llm.invoke(prompt)
    state.recommendations = response.content
    
    # Log token usage for monitoring
    logger.info(f"🔍 RECOMMENDATIONS_NODE usage_metadata: {response.usage_metadata}")
    
    return state
```

### **4.6 Create Graph Assembly Function**
```python
# Add to langgraph_workflow_function.py

def create_graph(config: LanggraphWorkflowFunctionConfig):
    """Create and configure the LangGraph workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("analyze", lambda state: analysis_node(state, config))
    workflow.add_node("recommend", lambda state: recommendations_node(state, config))
    
    # Define workflow edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", END)
    
    return workflow.compile()
```

---

## 5. AIQ Workflow Registration

### **5.1 Create Main Workflow Function**
```python
# Add to langgraph_workflow_function.py

async def get_workflow_response_function(config: LanggraphWorkflowFunctionConfig):
    """
    Main workflow function that returns the response function for execution
    """
    logger.info("get_workflow_response_function: Preparing response function.")

    @track_function
    async def _response_fn(input_message: str) -> str:
        """Internal response function that executes the workflow"""
        logger.info(f"_response_fn: Starting for input: {input_message}")
        
        # Create graph and execute
        graph = create_graph(config)
        initial_state = GraphState(topic=input_message)
        result = graph.invoke(initial_state)
        
        logger.info(f"📊 Graph execution completed with dedicated LLM instances per node")
        
        # Format comprehensive output
        output = f"""🚀 LangGraph Analysis Results for: {input_message}
{"=" * 60}

📊 PER-NODE TOKEN TRACKING (AIQ Native):
{"-" * 40}
✅ Each node uses dedicated LLM instance with AIQ tracking
📋 Token usage captured per node in logs
🔍 Detailed per-node metrics available for monitoring

📊 ANALYSIS:
{"-" * 40}
{result['analysis']}

💡 RECOMMENDATIONS:
{"-" * 40}
{result['recommendations']}

✅ Graph execution completed with per-node AIQ token tracking!"""
        
        logger.info("_response_fn: Workflow execution complete.")
        return output

    return _response_fn
```

### **5.2 Create Registration File**
```python
# File: langgraph_workflow/src/langgraph_workflow/register.py

# pylint: disable=unused-import
# flake8: noqa

# Import any tools which need to be automatically registered here
from langgraph_workflow import langgraph_workflow_function
```

### **5.3 Create Package Definition**
```toml
# File: langgraph_workflow/pyproject.toml

[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools >= 64"]

[project]
name = "langgraph_workflow"
version = "0.1.0"
dependencies = [
  "aiqtoolkit[langchain]",
]
requires-python = ">=3.11,<3.13"
description = "Custom AIQ Toolkit Workflow with LangGraph"
classifiers = ["Programming Language :: Python"]

[project.entry-points.'aiq.components']
langgraph_workflow = "langgraph_workflow.register"
```

### **5.4 Create AIQ Configuration**
```yaml
# File: langgraph_workflow/configs/config.yml

general:
  use_uvloop: true
  logging:
    console:
      _type: console
      level: INFO

  front_end:
    _type: console

  profiling:
    enabled: true
    export_format: file
    output_file: "token_profiling_results.json"

workflow:
  _type: langgraph_workflow
  llm_url: "https://integrate.api.nvidia.com/v1"
  llm_api_key: "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"
  llm_model_name: "nvidia/llama-3.1-nemotron-70b-instruct"
  temperature: 0.7
  first_node_prompt: "Analyze the following topic and provide key insights: {topic}"
  second_node_prompt: "Based on the analysis: {analysis}, provide detailed recommendations and action items."
```

---

## 6. OpenTelemetry Integration

### **6.1 Add OpenTelemetry Imports**
```python
# Add to top of langgraph_workflow_function.py

# OpenTelemetry imports for metrics
from opentelemetry import metrics
from opentelemetry.metrics import Counter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
```

### **6.2 Add OpenTelemetry Initialization**
```python
# Add to langgraph_workflow_function.py

# Global OTel instruments
_otel_meter_instruments: Dict[str, Counter] = {}
_otel_initialized = False

def _initialize_otel_in_workflow():
    """Initialize OpenTelemetry MeterProvider and Counters"""
    global _otel_initialized
    global _otel_meter_instruments

    if _otel_initialized:
        logger.info("OpenTelemetry already initialized. Skipping.")
        return

    logger.info("Initializing OpenTelemetry within workflow process.")

    # Create resource with service metadata
    resource = Resource.create({
        "service.name": "langgraph-aiq-workflow-nodes",
        "service.instance.id": os.getenv("HOSTNAME", "localhost"),
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })

    # Get OTLP endpoint from environment
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otel_endpoint:
        logger.error("OTEL_EXPORTER_OTLP_ENDPOINT environment variable not set!")
        return

    # Configure OTLP metric exporter
    metric_exporter = OTLPMetricExporter(
        endpoint=otel_endpoint,
        insecure=True
    )
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    logger.info(f"OpenTelemetry configured, sending to: {otel_endpoint}")

    # Create metrics instruments
    global_meter = metrics.get_meter(__name__)
    
    _otel_meter_instruments["llm_tokens_total"] = global_meter.create_counter(
        "llm_tokens_total", 
        description="Total tokens consumed by LLM operations (prompt + completion)", 
        unit="token"
    )
    _otel_meter_instruments["llm_prompt_tokens_total"] = global_meter.create_counter(
        "llm_prompt_tokens_total", 
        description="Total prompt tokens consumed by LLM operations", 
        unit="token"
    )
    _otel_meter_instruments["llm_completion_tokens_total"] = global_meter.create_counter(
        "llm_completion_tokens_total", 
        description="Total completion tokens consumed by LLM operations", 
        unit="token"
    )
    
    logger.info(f"OpenTelemetry metrics instruments created: {list(_otel_meter_instruments.keys())}")
    _otel_initialized = True
```

### **6.3 Add Metrics Emission to Nodes**
```python
# Update analysis_node function

@track_function
def analysis_node(state: GraphState, config: LanggraphWorkflowFunctionConfig) -> GraphState:
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
        logger.warning(f"⚠️ OTel metrics NOT emitted for analysis_node.")
    
    return state

# Update recommendations_node function similarly...
```

### **6.4 Update Main Workflow Function**
```python
# Update get_workflow_response_function

async def get_workflow_response_function(config: LanggraphWorkflowFunctionConfig):
    """Main workflow function with OpenTelemetry initialization"""
    logger.info("get_workflow_response_function: Initializing OTel and preparing response function.")
    
    # Initialize OpenTelemetry when workflow starts
    _initialize_otel_in_workflow()

    @track_function
    async def _response_fn(input_message: str) -> str:
        # ... existing workflow logic ...
        
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
        
        return output

    return _response_fn
```

---

## 7. Docker Configuration

### **7.1 Create Application Dockerfile**
```dockerfile
# File: Dockerfile

# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Set entrypoint to the telemetry-enabled script
ENTRYPOINT ["python", "run_with_json_export_tel.py"]

# Default command with example input
CMD ["AI safety measures in autonomous vehicles containerized"]
```

### **7.2 Create Main Execution Script**
```python
# File: run_with_json_export_tel.py

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import workflow components
from langgraph_workflow.src.langgraph_workflow.langgraph_workflow_function import (
    get_workflow_response_function, 
    LanggraphWorkflowFunctionConfig
)

async def main_workflow_run(input_message: str):
    """Initialize and run the AIQ workflow"""
    logger.info("Starting containerized LangGraph AIQ workflow.")

    # Create workflow configuration
    workflow_config = LanggraphWorkflowFunctionConfig()

    # Get response function from workflow
    _response_fn = await get_workflow_response_function(config=workflow_config)

    logger.info(f"Running workflow with input: {input_message}")
    workflow_result = await _response_fn(input_message)
    logger.info("Workflow execution completed.")
    
    return workflow_result

if __name__ == "__main__":
    logger.info("Script started in __main__ block.")
    
    # Get input message from command line or use default
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
    time.sleep(5)  # Allow time for metrics export
    logger.info("Exiting script.")
```

### **7.3 Create Environment File**
```bash
# File: .env

NVIDIA_API_KEY='your key'
```

---

## 8. Monitoring Stack Setup

### **8.1 Create OpenTelemetry Collector Configuration**
```yaml
# File: otel-collector-config.yaml

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
  debug:
    verbosity: detailed

service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus, debug]
    traces:
      receivers: [otlp]
      exporters: [debug]
```

### **8.2 Create Prometheus Configuration**
```yaml
# File: prometheus.yml

global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']
```

### **8.3 Create Docker Compose Configuration**
```yaml
# File: docker-compose.yml

services:
  # OpenTelemetry Collector - Receives and exports metrics
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: otel-collector
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otelcol/config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver  
      - "8889:8889"   # Prometheus exporter endpoint
    restart: unless-stopped

  # Prometheus - Stores and queries metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    command: ["--config.file=/etc/prometheus/prometheus.yml", "--web.listen-address=:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"   # Prometheus UI
    depends_on:
      - otel-collector
    restart: unless-stopped

  # Grafana - Visualizes metrics  
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"   # Grafana UI
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

  # LangGraph Application - Generates metrics
  langgraph-app:
    build: .
    container_name: langgraph-app
    environment:
      # Pass NVIDIA API Key
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      # Configure OpenTelemetry endpoint
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
    depends_on:
      - otel-collector

# Persistent storage for monitoring data
volumes:
  prometheus_data:
  grafana_data:
```

---

## 9. Testing & Validation

### **9.1 Build and Start Services**
```bash
# Start monitoring stack
docker-compose up -d

# Check service status
docker-compose ps
```

### **9.2 Run Workflow with Monitoring**
```bash
# Run workflow and generate metrics
docker-compose run --rm langgraph-app "blockchain security analysis"

# Run multiple times to accumulate metrics
docker-compose run --rm langgraph-app "AI ethics in healthcare"
docker-compose run --rm langgraph-app "quantum computing applications"
```

### **9.3 Validate Metrics Collection**
```bash
# Check raw metrics from collector
curl http://localhost:8889/metrics | grep llm_

# Test Prometheus API
curl "http://localhost:9090/api/v1/query?query=llm_tokens_token_total"

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### **9.4 Access Monitoring UIs**

**Prometheus UI:**
1. Open: http://localhost:9090
2. Query: `llm_tokens_token_total`
3. Click "Execute" and check "Table" tab

**Grafana Dashboard:**
1. Open: http://localhost:3000  
2. Login: `admin` / `admin`
3. Add Prometheus data source: `http://prometheus:9090`
4. Create dashboard with token metrics

---

## 10. Production Deployment

### **10.1 Environment Configuration**
```yaml
# File: docker-compose.prod.yml

version: '3.8'
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  prometheus:
    image: prom/prometheus:latest
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    volumes:
      - prometheus_data:/prometheus
    configs:
      - source: prometheus_config
        target: /etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_password_here
      - GF_USERS_ALLOW_SIGN_UP=false

  langgraph-app:
    build: .
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - ENVIRONMENT=production

configs:
  prometheus_config:
    file: ./prometheus.yml

volumes:
  prometheus_data:
    driver: local
```

### **10.2 Security Hardening**
```bash
# Create production secrets
docker secret create nvidia_api_key /path/to/api_key.txt

# Enable Grafana authentication
docker-compose exec grafana grafana-cli admin reset-admin-password NewSecurePassword

# Configure Prometheus retention
# Add to prometheus.yml:
# global:
#   external_labels:
#     environment: 'production'
# rule_files:
#   - "alert_rules.yml"
```

### **10.3 Monitoring & Alerting**
```yaml
# File: alert_rules.yml

groups:
  - name: langgraph_alerts
    rules:
      - alert: HighTokenUsage
        expr: sum(rate(llm_tokens_total[5m])) > 1000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High token usage detected"
          description: "Token usage is {{ $value }} tokens/minute"
      
      - alert: WorkflowFailures
        expr: sum(rate(workflow_errors_total[5m])) > 5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High workflow failure rate"
```

### **10.4 Scaling Considerations**
```text
HORIZONTAL SCALING:
==================

1. APPLICATION LAYER:
   - Multiple langgraph-app containers
   - Load balancer for request distribution
   - Kubernetes deployment for auto-scaling

2. MONITORING LAYER:
   - Prometheus federation for multi-cluster
   - Grafana clustering for high availability
   - InfluxDB for long-term storage

3. RESOURCE MANAGEMENT:
   - CPU/Memory limits per container
   - Persistent volumes for data
   - Network policies for security

4. OPERATIONAL:
   - Health checks and liveness probes
   - Log aggregation (ELK/Loki)
   - Backup and disaster recovery
```

---

## 🎯 Summary

You've now built a complete production-ready system with:

✅ **LangGraph Workflow** - Two-node AI analysis pipeline
✅ **AIQ Integration** - Native profiling and token tracking  
✅ **OpenTelemetry** - Comprehensive metrics collection
✅ **Docker Compose** - Multi-service orchestration
✅ **Prometheus** - Metrics storage and querying
✅ **Grafana** - Dashboard visualization
✅ **Production Ready** - Security, scaling, monitoring

**Next Steps:**
1. Customize prompts for your use case
2. Add more workflow nodes as needed
3. Create custom Grafana dashboards
4. Set up alerting rules
5. Deploy to production environment

**🚀 Your containerized LangGraph with full observability is ready!** 