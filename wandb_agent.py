#!/usr/bin/env python3
"""
Ollama + LangGraph + W&B observability example
Includes LLM performance metrics and system metrics per query.
"""

import os
import time
import wandb
import psutil
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# Optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_available = True
except ImportError:
    gpu_available = False

# -----------------------------
# 1. Setup Weights & Biases
# -----------------------------
os.environ["WANDB_PROJECT"] = "ollama-langgraph-demo"

if not wandb.run:
    wandb.init(
        project="ollama-langgraph-demo",
        name="prompt-observability-demo",
        config={
            "model": "mistral-nemo",
            "temperature": 0,
            "framework": "langgraph",
            "max_tokens": 500
        }
    )

# Define numeric summaries
wandb.define_metric("performance/latency_seconds", summary="mean")
wandb.define_metric("performance/tokens_per_second", summary="mean")
wandb.define_metric("tokens/cumulative_total", summary="max")
wandb.define_metric("system/cpu_usage_percent", summary="max")
wandb.define_metric("system/ram_usage_percent", summary="max")
wandb.define_metric("system/gpu_utilization", summary="max")
wandb.define_metric("system/gpu_memory_used", summary="max")

# -----------------------------
# 2. Configure Ollama LLM
# -----------------------------
llm = ChatOllama(
    model="mistral-nemo",
    temperature=0,
    num_predict=500,
    top_k=40,
    top_p=0.9
)

# -----------------------------
# 3. System Prompt
# -----------------------------
system_prompt = "You are a helpful assistant that provides clear, concise answers."

# -----------------------------
# 4. Create Agent with LangGraph
# -----------------------------
agent_executor = create_react_agent(
    model=llm,
    tools=[],
    prompt=system_prompt
)

# -----------------------------
# 5. Token Estimation
# -----------------------------
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

# -----------------------------
# 6. System Metrics Logging
# -----------------------------
def log_system_metrics(step):
    metrics = {
        "step": step,
        "system/cpu_usage_percent": psutil.cpu_percent(),
        "system/ram_usage_percent": psutil.virtual_memory().percent
    }
    if gpu_available:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics.update({
            "system/gpu_memory_used": mem_info.used / (1024**2),  # MB
            "system/gpu_utilization": util.gpu
        })
    wandb.log(metrics)

# -----------------------------
# 7. Run Queries with Logging
# -----------------------------
if __name__ == "__main__":
    test_queries = [
        "Tell me a short joke about artificial intelligence.",
        "What is the capital of France?",
        "Explain quantum computing in one sentence."
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_duration = 0

    for step, user_input in enumerate(test_queries, start=1):
        print(f"\n{'='*50}\nQuery {step}: {user_input}\n{'='*50}")

        input_tokens = estimate_tokens(user_input + system_prompt)
        start_time = time.time()

        try:
            # Invoke agent
            response = agent_executor.invoke({"messages": [("user", user_input)]})
            end_time = time.time()
            duration = end_time - start_time

            assistant_reply = response['messages'][-1].content
            output_tokens = estimate_tokens(assistant_reply)
            total_tokens = input_tokens + output_tokens

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_duration += duration

            print(f"\nAssistant: {assistant_reply}")

            # Log all metrics including system metrics
            wandb.log({
                "step": step,
                "query_number": step,
                "success": True,

                # Token metrics
                "tokens/estimated_input": input_tokens,
                "tokens/estimated_output": output_tokens,
                "tokens/estimated_total": total_tokens,
                "tokens/cumulative_input": total_input_tokens,
                "tokens/cumulative_output": total_output_tokens,
                "tokens/cumulative_total": total_input_tokens + total_output_tokens,

                # Performance metrics
                "performance/latency_seconds": duration,
                "performance/tokens_per_second": total_tokens / duration if duration > 0 else 0,

                # Model info
                "model": "mistral-nemo",
                "temperature": 0,

                # Text logging
                "user_input_text": wandb.Html(user_input),
                "assistant_reply_text": wandb.Html(assistant_reply),

                "timestamp": end_time
            })

            # System metrics
            log_system_metrics(step)

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"\nError: {e}")
            wandb.log({
                "step": step,
                "query_number": step,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "performance/latency_seconds": duration,
                "timestamp": end_time
            })
            log_system_metrics(step)

    # Final session summary
    total_tokens_processed = total_input_tokens + total_output_tokens
    avg_latency = total_duration / len(test_queries)
    wandb.log({
        "step": step + 1,
        "total_queries": len(test_queries),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens_processed": total_tokens_processed,
        "total_session_duration": total_duration,
        "average_latency_seconds": avg_latency,
        "average_tokens_per_query": total_tokens_processed / len(test_queries),
        "overall_tokens_per_second": total_tokens_processed / total_duration if total_duration > 0 else 0,
        "session_complete": True
    })

print("\n🎉 Done! Check your W&B dashboard for LLM + system observability metrics.")
