# 📊 Monitoring Setup Guide

## ✅ **Current Status: WORKING!**

Your Prometheus and Grafana logging setup with Docker Compose is **fully functional**. Here's what's working:

### 🎯 **Successfully Working Components:**

1. **✅ OpenTelemetry Collector** - Receiving and exporting metrics
2. **✅ Prometheus** - Scraping and storing metrics  
3. **✅ Grafana** - Ready for dashboard creation
4. **✅ LangGraph Application** - Emitting per-node token metrics
5. **✅ Metrics Export** - JSON and Prometheus format

---

## 🔗 **Access URLs**

| Service | URL | Description |
|---------|-----|-------------|
| **OTel Collector Metrics** | http://localhost:8889/metrics | Raw Prometheus metrics from collector |
| **Prometheus UI** | http://localhost:9090 | Query and explore metrics |
| **Prometheus API** | http://localhost:9090/api/v1/query | Programmatic metric queries |
| **Grafana Dashboard** | http://localhost:3000 | Create dashboards (admin/admin) |

---

## 📊 **Available Metrics**

Your application is successfully emitting these metrics:

### **Token Usage Metrics:**
```prometheus
# Total tokens per node
llm_tokens_token_total{node_name="analysis_node"} 711
llm_tokens_token_total{node_name="recommendations_node"} 1389

# Prompt tokens per node  
llm_prompt_tokens_token_total{node_name="analysis_node"} 29
llm_prompt_tokens_token_total{node_name="recommendations_node"} 667

# Completion tokens per node
llm_completion_tokens_token_total{node_name="analysis_node"} 682  
llm_completion_tokens_token_total{node_name="recommendations_node"} 722
```

### **Labels Available:**
- `node_name`: "analysis_node" or "recommendations_node"
- `llm_model`: "nvidia/llama-3.1-nemotron-70b-instruct"
- `instance`: Container ID
- `job`: "langgraph-aiq-workflow-nodes"

---

## 🚀 **How to Use**

### **1. View Raw Metrics**
```bash
curl http://localhost:8889/metrics | grep llm_
```

### **2. Query with Prometheus**
```bash
# Total tokens by node
curl "http://localhost:9090/api/v1/query?query=llm_tokens_token_total"

# Sum all tokens
curl "http://localhost:9090/api/v1/query?query=sum(llm_tokens_token_total)"
```

### **3. Useful Prometheus Queries**
```promql
# Total tokens consumed
sum(llm_tokens_token_total)

# Tokens by node
llm_tokens_token_total

# Token rate over time
rate(llm_tokens_token_total[5m])

# Cost estimation (example: $0.01 per 1000 tokens)
sum(llm_tokens_token_total) * 0.00001
```

### **4. Create Grafana Dashboard**
1. Go to http://localhost:3000
2. Login: `admin` / `admin`
3. Add Data Source: Prometheus (http://prometheus:9090)
4. Create dashboard with queries above

---

## 🐳 **Docker Commands**

### **Start/Stop Services**
```bash
# Start all services
docker-compose up -d

# Stop all services  
docker-compose down

# Rebuild app with changes
docker-compose up --build -d langgraph-app

# View logs
docker-compose logs langgraph-app
docker-compose logs prometheus
docker-compose logs otel-collector
```

### **Run New Workflow Instance**
```bash
# Run with custom input
docker-compose run --rm langgraph-app "Your custom topic here"
```

---

## 📈 **Sample Grafana Dashboard Panels**

### **Panel 1: Total Token Usage**
- **Query:** `sum(llm_tokens_token_total)`
- **Type:** Stat
- **Title:** "Total Tokens Consumed"

### **Panel 2: Tokens by Node**
- **Query:** `llm_tokens_token_total`
- **Type:** Bar Chart
- **Title:** "Token Usage by Node"

### **Panel 3: Token Usage Over Time**
- **Query:** `increase(llm_tokens_token_total[1h])`
- **Type:** Time Series
- **Title:** "Token Consumption Rate"

### **Panel 4: Cost Tracking**
- **Query:** `sum(llm_tokens_token_total) * 0.00001`
- **Type:** Stat  
- **Title:** "Estimated Cost ($)"

---

## 🔧 **Troubleshooting**

### **Issue: No Metrics Appearing**
```bash
# Check if collector is receiving data
docker-compose logs otel-collector | grep -i error

# Check if app is running
docker-compose logs langgraph-app

# Verify metrics endpoint
curl -s http://localhost:8889/metrics | grep llm_ | wc -l
```

### **Issue: Prometheus Not Scraping**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Prometheus config
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml
```

### **Issue: Container Failures**
```bash
# Check service status
docker-compose ps

# Restart specific service
docker-compose restart otel-collector
docker-compose restart prometheus
```

---

## 🎯 **Next Steps**

1. **✅ Complete** - Basic monitoring setup
2. **🚀 Recommended** - Create Grafana dashboards
3. **📊 Optional** - Add alerting rules
4. **🔄 Optional** - Set up persistent storage
5. **🌐 Optional** - Add authentication

---

## 📁 **Configuration Files**

Your working configuration files:

- **Docker Compose:** `docker-compose.yml` ✅
- **OTel Collector:** `otel-collector-config.yaml` ✅  
- **Prometheus:** `prometheus.yml` ✅
- **Application:** `run_with_json_export_tel.py` ✅
- **Workflow:** `langgraph_workflow_function.py` ✅

**🎉 Your monitoring stack is fully operational!** 