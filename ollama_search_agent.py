from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
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

tool = Tool(
    name="duckduckgo_search",
    description="Use this tool to search for real-time or recent information on the web.",
    func=search.run
)

print(type(tool))
print(tool.name)

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
                    # DuckDuckGo tool expects just a string input
                    if isinstance(tool_args, dict) and "__arg1" in tool_args:
                        input_str = tool_args["__arg1"]
                    else:
                        input_str = str(tool_args)

                    response = self.tools[tool_name].invoke(input_str)
                except Exception as e:
                    response = f" Error during tool execution: {str(e)}"

            # Append result as ToolMessage
            results.append(
                ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(response))
            )

        print("Tool Results:", results)
        return {'messages': results}




# Prompt suitable for REACT loop
prompt =  """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

# Use Ollama model (instruct-tuned)

try:
    # Initialize the model
    model = ChatOllama(model="mistral-nemo")
    # Send a simple test prompt
    response = model.invoke("Hello!")
    print("✅ Model is reachable. Response:", response)
except Exception as e:
    print("❌ Could not reach the model. Error:", str(e))



# Initialize the agent with DuckDuckGo
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
query = "What is the best telecom operator in the UK? How many branches does it have?"
# Example user message
messages = [HumanMessage(content=query)]
thread = {"configurable": {"thread_id": "1"}}
state = {"messages": messages}
for event in abot.graph.stream(state, thread):
    for v in event.values():
        print(v)
        print(f'The next node is: {abot.graph.get_state(thread).next}')
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("Do you want to proceed to the next node? (y/n):  ")
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

