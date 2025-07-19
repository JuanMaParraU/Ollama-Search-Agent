import time
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import wikipedia
import os
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
#from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4


os.environ["NO_PROXY"] = "127.0.0.1,localhost"
memory = MemorySaver()
#memory = SqliteSaver.from_conn_string(":memory:")

# Define DuckDuckGo tool manually using the API wrapper
search = DuckDuckGoSearchAPIWrapper()

# Define a function to search Wikipedia
# This function will be used as a tool in the agent
def wikipedia_search(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

# Define the DuckDuckGo search tool
duck_tool = Tool(
    name="duckduckgo_search",
    description="Use this tool to search for real-time or recent information on the web.",
    func=search.run
)
# Define the Wikipedia search tool
wiki_tool = Tool(
    name="wikipedia_search",
    description="Use this tool to search Wikipedia for factual information.",
    func=wikipedia_search
)

print(type(duck_tool))
print(duck_tool.name)

def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

# Function to rewrite the query using the LLM
# This function will be called before using the Wikipedia search tool
# It rewrites the original query into a concise Wikipedia article title or keyword phrase
def rewrite_query_with_llm(original_query: str) -> str:
    print(f"Rewriting query>>>>>>>>")
    # Initialize the model
    model = ChatOllama(model="mistral-nemo")
    prompt = (
        "Rewrite the following user search query into a concise Wikipedia article title or keyword phrase.\n"
        "Respond with ONLY the rewritten title or phrase on a single line, with no explanations.\n\n"
        "Example:\n"
        "- 'best telecom operator UK' → Telecommunications in the United Kingdom\n"
        f"Input query:\n{original_query}\n\n"
        "Rewritten query:"
    )
    response = model.invoke(prompt)
    rewritten = response.content.strip()
    if not rewritten or rewritten.lower() == original_query.lower():
        return original_query
    return rewritten

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]

# Agent class using REACT-style reasoning loop
class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )

        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

    def call_model(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls
        results = []

        for t in tool_calls:
            tool_name = t['name']
            tool_args = t['args']

            print(f"🔧 Tool: {tool_name} | Args: {tool_args}")

            if tool_name not in self.tools:
                print(f"Tool '{tool_name}' not found!")
                result = "Invalid tool name"
            else:
                try:
                    input_str = tool_args.get("__arg1", str(tool_args))
                    response = self.tools[tool_name].invoke(input_str)
                except Exception as e:
                    print("⚠️ First tool failed:", str(e))
                    response = f"Failed: {str(e)}"

                    # 🔁 Fallback to Wikipedia if DuckDuckGo fails
                    if tool_name == "duckduckgo_search":
                        print(">> Fallback to Wikipedia")
                        wiki_tool = self.tools.get("wikipedia_search")
                        if wiki_tool:
                            try:
                                # Rewrite query before calling Wikipedia
                                rewritten_query = rewrite_query_with_llm(input_str)
                                print(f"Rewritten query for Wikipedia: {rewritten_query}")
                                response = wiki_tool.invoke(rewritten_query)
                                tool_name = "wikipedia_search"
                            except Exception as e2:
                                response = f"Wikipedia fallback also failed: {str(e2)}"

            results.append(
                ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(response))
            )

        print("Tool Results:", results)
        return {'messages': results}
    




# Prompt suitable for REACT loop
system_prompt = """You are a smart research assistant. Use the tools provided to retrieve accurate and up-to-date information. \
You may make multiple tool calls to gather sufficient information.

Instructions:
1. Start by using the 'duckduckgo_search' tool to search the web.
2. If 'duckduckgo_search' fails due to a rate limit, network error, or no results:
    - Fall back to the 'wikipedia_search' tool.
    - Before calling 'wikipedia_search', you MUST rewrite the query into a likely Wikipedia article title or keyword.
      Examples:
        - 'best telecom operator UK' → 'Telecommunications in the United Kingdom'
        - 'top AI companies' → 'Artificial intelligence companies'
        - 'famous UK poets' → 'British poets'
3. Do NOT reuse the original DuckDuckGo query for Wikipedia.
4. If both tools fail, ask the user to rephrase their question or specify what they want to know.
5. Prioritize tool-sourced results over internal knowledge.
6. ALWAYS mention which tool or source was used to obtain the result.

Return a concise and factual summary. Your job is to research using tools—not to guess."""


# Use Ollama model (instruct-tuned)

try:
    # Initialize the model
    model = ChatOllama(model="mistral-nemo")
    # Send a simple test prompt
    response = model.invoke("Hello!")
    print("✅ Model is reachable. Response:", response.content)
    model_name = response.response_metadata.get('model_name')
    print("Model name:", model_name)
    print("\n")
except Exception as e:
    print("❌ Could not reach the model. Error:", str(e))


# Initialize the agent with DuckDuckGo
abot = Agent(model, [duck_tool, wiki_tool], system=system_prompt, checkpointer=memory)
query = "What is the best telecom operator in the UK? How many branches does it have?"
# Example user message
messages = [HumanMessage(content=query)]
thread = {"configurable": {"thread_id": "1"}}
state = {"messages": messages}
n = 0
for event in abot.graph.stream(state, thread):
    for v in event.values():
        if isinstance(v, dict) and 'messages' in v:
            print(v['messages'])
print(f'The next node is: {abot.graph.get_state(thread).next}')

while abot.graph.get_state(thread).next:
    #print("\n", abot.graph.get_state(thread),"\n")
    _input = input(f"Do you want to proceed to the next node {abot.graph.get_state(thread).next}? (y/n):  ")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)

# Get the final state
final_state = abot.graph.get_state(thread)
final_messages = final_state.values["messages"]

# Find the last AIMessage
final_ai_message = next((msg for msg in reversed(final_messages) if isinstance(msg, AIMessage)), None)

if final_ai_message:
    print("🧠 Final response from the agent:")
    print(final_ai_message.content)
else:
    print("⚠️ No AI response found in the final state.")

