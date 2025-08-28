import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

async def init_agent():
    client = MultiServerMCPClient(
        {
            "research": {
                "url": "http://localhost:8000/mcp/",  # ✅ Port matches your server
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()
    print("\n🔧 Available Tools:", [tool.name for tool in tools])
    # Use Ollama model (instruct-tuned)

    try:
        # Initialize the model
        model = ChatOllama(model="mistral")
        # Send a simple test prompt
        response = model.invoke("Hello!")
        print("✅ Model is reachable. Response:", response.content)
        model_name = response.response_metadata.get('model_name')
        print("Model name:", model_name)
        print("\n")
    except Exception as e:
        print("❌ Could not reach the model. Error:", str(e))

    # Create a React agent with the model and tools. This will allow the agent to use tools as needed. 
    # The agent will automatically handle tool calls and responses.
    # It will also handle the conversation history and context.
    # The agent will use the tools to answer the query.
    # The agent will also handle the conversation history and context.
    # It is langgraph's prebuilt React agent, which is designed to handle tool calls and responses.
    # The manual graph building and tool call handling developed in the main repo is abstracted by the create_react_agent function.
    system_prompt = """
    You are a smart research assistant that uses tools to find information. You can make multiple tool calls before answering if needed.
    Available tools:
    - 'duckduckgo_search' — for general web search.
    - 'wikipedia_search' — for encyclopedia-style lookups.
    """
    config = {
    "configurable": {
        "thread_id": "research-session-1",  # Required
        "checkpoint_ns": "",
        "checkpoint_id": "research-session-1",  # Required
        }
    }
    agent = create_react_agent( model, 
                                tools, 
                                debug=True,
                                prompt=system_prompt, 
                                checkpointer=memory)

    return agent, config

# Synchronous helper to call from FastAPI
def create_agent():
    return asyncio.run(init_agent())
