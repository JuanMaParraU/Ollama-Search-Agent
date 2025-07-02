
# 🧠 LangGraph AI Agent with Ollama & DuckDuckGo Search

This project demonstrates a **REACT-style AI research assistant** built with **LangGraph**, **LangChain**, and **Ollama**. The agent is capable of reasoning over user input, calling real-time tools like **DuckDuckGo Search**, and maintaining conversational state across turns.

It uses a **looping LLM-agent pattern** where the LLM decides when to use tools, collects information, and responds after reasoning over the results. This code has been adapted from the course **AI Agents in LangGraph** from [DeepLearning.AI](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/qyrpc/introduction).

---

## 📦 Requirements and Dependencies

Basic Requirements
- **Python 3.10+**
- **Git**

#### Clone this repo:
```bash
git clone git@github.com:JuanMaParraU/Ollama-Search-Agent.git
```

#### Create a virtual environment
```bash
python -m venv myenv
```
#### Activate the environment
On macOS/Linux:
```bash
source myenv/bin/activate
```

On Windows:
```bash
myenv\Scripts\activate
````

#### Install the required packages:

```bash
pip install langgraph langchain langchain-community langchain-core langchain-ollama duckduckgo-search
```
or 

```bash
pip install -r requirements.txt
```

Also required
- **Ollama** (local LLM server)

---

## 🦙 Why Use Ollama?

[**Ollama**](https://ollama.com) allows you to run powerful LLMs **locally** on your machine. This reduces latency and eliminates external API costs.

In this project, we use the **`mistral-nemo`** instruct-tuned model via Ollama, which supports structured tool calls and high-quality generation.

### 🔧 Installing Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the server:

```bash
ollama serve
```

### 📥 Download the Model

To download the `mistral-nemo` model:

```bash
ollama run mistral-nemo
```

---

## 🔄 Agent–Model Communication Diagram

```plaintext
User Query
   ↓
[HumanMessage]
   ↓
[llm Node] ──▶ Model (Ollama: Mistral-nemo)
   │            │
   │         ┌──┴────────────┐
   │         │Tool Call(s)?  │
   ▼         └──────┬────────┘
[Action Node]       │
   │                ▼
   └──── DuckDuckGo Tool ──▶ Response
   ▼
[ToolMessage] → Back to [llm Node] (loop)
   ↓
[AIMessage] Final Response
```

---

## 👤 Human-in-the-Loop (HITL) Interrupt

This agent is designed with **human oversight in mind**, thanks to **LangGraph's interrupt mechanism**.

Before the agent executes any tool-based action (e.g., calling DuckDuckGo), it **pauses and prompts the user** for confirmation. This gives users full control over whether to allow the action or stop the reasoning loop.

### 🔄 When Does It Pause?

The pause occurs before the `"action"` node executes — controlled by:

```python
interrupt_before=["action"]
```

### 🧠 Why Use This?

- Ensures transparency of model behavior.
- Prevents unintended tool usage.
- Allows debugging or exploring intermediate LLM reasoning.

### 🔍 Example Flow

```
User Query → [LLM Node] → Suggests DuckDuckGo Tool
       ↓
🛑 Prompt: Do you want to proceed to the next node? (y/n):
```

You can inspect the tool arguments before confirming, offering a clear **"human-in-the-loop" control point**.

---

## 🚀 Running the Code

After setting up dependencies and Ollama, run the script directly:

```bash
python ollama_search_agent.py
```

You’ll be prompted whether to continue each loop (i.e., tool invocation and model reasoning). This enables step-by-step exploration.

Example user query:
```
What is the best telecom operator in the UK? How many branches does it have?
```

---

## 🔍 Main Components

### 1. **Agent Class**

- Implements a **REACT loop** using LangGraph.
- Decides whether to invoke tools based on model output.
- Manages message state with merging logic (`reduce_messages`).
- Uses nodes: `llm`, `action`, and `END`.

### 2. **Tool Integration**

- Uses `Tool` abstraction to wrap `DuckDuckGoSearchAPIWrapper`.
- Only string inputs supported (transformed from model arguments).
- Tool results are converted to `ToolMessage` and injected into the loop.

### 3. **Model Invocation**

- `ChatOllama` wraps the local Ollama model.
- `bind_tools()` enables structured tool usage by the LLM.

### 4. **Prompt (REACT-style)**

Located near the bottom:

```python
system_prompt = """You are a smart research assistant. Use the search engine to look up information..."""
```

This prompt encourages the LLM to think step-by-step, calling tools when appropriate.

---

## 🧪 Example Output

```
🔧 Tool: duckduckgo_search | Args: {'__arg1': 'best telecom operator in UK'}
Tool Results: [ToolMessage(...)]
The next node is: llm
...
🧠 Final response from the agent:
"Based on recent results, the top telecom operator in the UK is..."
```

---

## ✏️ How to Update the Prompts

1. Open the `prompt = """..."""` block.
2. Modify the system instructions to match your use case.
3. You can make the assistant more cautious, curious, or specific depending on the task domain.

For example:

```python
prompt = """You are a legal research assistant. Search for the latest regulatory updates..."""
```

Also, adjust the `SystemMessage(content=self.system)` injection inside `Agent.call_model()` for context-aware runs.

---

## 🧾 License

Apache 2.0 
