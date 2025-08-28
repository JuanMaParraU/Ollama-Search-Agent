# minimal_agent_api_clean_final.py
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from callable_react_agent import create_agent

app = FastAPI()

# Enable CORS for local UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent, config = create_agent()
threads = {}

# ---------------- Validation ----------------
def validate_sse_payload(payload: dict):
    if "values" not in payload or not isinstance(payload["values"], list):
        raise ValueError("Payload must have 'values' as a list")
    for v in payload["values"]:
        if "delta" in v:
            content = v["delta"].get("content")
            if not isinstance(content, list):
                raise ValueError("'delta.content' must be a list of strings")
            for c in content:
                if not isinstance(c, str):
                    raise ValueError("Each 'delta.content' item must be a string")
        elif "messages" in v:
            messages = v["messages"]
            if not isinstance(messages, list):
                raise ValueError("'messages' must be a list")
            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    raise ValueError("Each message must have 'role' and 'content'")
                if not isinstance(msg["content"], list):
                    raise ValueError("'content' must be a list of text objects")
                for c in msg["content"]:
                    if not isinstance(c, dict) or "type" not in c or "text" not in c:
                        raise ValueError("Each content item must be {'type':'text','text':...}")
    return True

# ---------------- Helper ----------------
def message_to_dict(msg):
    """Convert HumanMessage, AIMessage, ToolMessage to frontend dict."""
    # AIMessage or HumanMessage
    if hasattr(msg, "content"):
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        content = msg.content
        if not isinstance(content, str):
            content = str(content)
        return {"role": role, "content": [{"type": "text", "text": content}]}
    # ToolMessage
    if hasattr(msg, "name") and hasattr(msg, "content"):
        return {"role": "tool", "content": [{"type": "text", "text": msg.content}]}
    # Fallback
    return {"role": "assistant", "content": [{"type": "text", "text": str(msg)}]}

# ---------------- API Endpoints ----------------
@app.get("/info")
async def info():
    return {"assistant_id": "agent", "name": "Agent"}

@app.post("/threads")
async def create_thread():
    thread_id = f"thread-{len(threads)+1}"
    checkpoint_id = f"checkpoint-{thread_id}-0"
    threads[thread_id] = {
        "messages": [],
        "checkpoint_id": checkpoint_id
    }
    return {
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id
    }

@app.post("/threads/search")
async def search_threads():
    # Return array directly, not wrapped in object
    result = []
    for tid in threads.keys():
        thread_data = threads[tid]
        checkpoint_id = thread_data.get("checkpoint_id", f"checkpoint-{tid}-0") if isinstance(thread_data, dict) else f"checkpoint-{tid}-0"
        result.append({
            "thread_id": tid,
            "checkpoint_id": checkpoint_id
        })
    return result

@app.post("/threads/{thread_id}/history")
async def get_history(thread_id: str):
    thread_data = threads.get(thread_id, {"messages": [], "checkpoint_id": f"checkpoint-{thread_id}-0"})
    
    # Handle both old format (list) and new format (dict)
    if isinstance(thread_data, list):
        messages = thread_data
        checkpoint_id = f"checkpoint-{thread_id}-0"
        # Migrate to new format
        threads[thread_id] = {
            "messages": messages,
            "checkpoint_id": checkpoint_id
        }
    else:
        messages = thread_data.get("messages", [])
        checkpoint_id = thread_data.get("checkpoint_id", f"checkpoint-{thread_id}-0")
    
    # Return array directly, not wrapped in {"values": ...}
    result = []
    for i, msg in enumerate(messages):
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        
        # Each history item needs checkpoint_id
        result.append({
            "messages": [{"role": msg["role"], "content": content}],
            "checkpoint_id": f"checkpoint-{thread_id}-{i+1}"
        })
    
    return result

@app.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: Request):
    data = await request.json()

    # ---------------- Extract user message ----------------
    def extract_user_message(payload: dict) -> str:
        if "input" in payload and "messages" in payload["input"]:
            messages = payload["input"]["messages"]
            if messages:
                last_msg = messages[-1]
                if "content" in last_msg and last_msg["content"]:
                    for item in last_msg["content"]:
                        if item.get("type") == "text":
                            return item.get("text", "")
        if "messages" in payload and payload["messages"]:
            last_msg = payload["messages"][-1]
            content = last_msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
        if "content" in payload and isinstance(payload["content"], str):
            return payload["content"]
        if "input" in payload and isinstance(payload["input"], str):
            return payload["input"]
        return "Hello"

    user_message = extract_user_message(data)

    if thread_id not in threads:
        threads[thread_id] = {
            "messages": [],
            "checkpoint_id": f"checkpoint-{thread_id}-0"
        }
    
    # Handle both old format (list) and new format (dict)
    thread_data = threads[thread_id]
    if isinstance(thread_data, list):
        messages = thread_data
        checkpoint_id = f"checkpoint-{thread_id}-0"
        threads[thread_id] = {
            "messages": messages,
            "checkpoint_id": checkpoint_id
        }
        thread_data = threads[thread_id]
    
    thread_data["messages"].append({"role": "user", "content": user_message})
    current_checkpoint = len(thread_data["messages"])

    # ---------------- Stream generator ----------------
    async def generate():
        full_response = ""
        
        try:
            # Send initial event to indicate streaming has started
            yield f"data: {json.dumps({'event': 'thread.run.created', 'run_id': 'run-123', 'checkpoint_id': f'checkpoint-{thread_id}-{current_checkpoint}'})}\n\n"
            
            # Small delay to ensure proper event ordering
            await asyncio.sleep(0.01)
            
            streamed_content = False
            
            # Try streaming first
            async for chunk in agent.astream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config
            ):
                if isinstance(chunk, dict) and "agent" in chunk:
                    agent_data = chunk["agent"]
                    if "messages" in agent_data:
                        for msg in agent_data["messages"]:
                            if isinstance(msg, AIMessage) and msg.content:
                                streamed_content = True
                                content_text = msg.content  # Use the full content
                                full_response = content_text  # Store full response
                                
                                # Send as complete message, not delta
                                msg_dict = message_to_dict(msg)
                                msg_payload = {
                                    "values": [{
                                        "messages": [msg_dict],
                                        "checkpoint_id": f"checkpoint-{thread_id}-{current_checkpoint + 1}"
                                    }]
                                }
                                validate_sse_payload(msg_payload)
                                yield f"data: {json.dumps(msg_payload)}\n\n"

            # If nothing was streamed, fall back to invoke
            if not streamed_content:
                try:
                    final_result = await agent.ainvoke(
                        {"messages": [HumanMessage(content=user_message)]},
                        config=config
                    )
                    
                    if "messages" in final_result:
                        for msg in final_result["messages"]:
                            if isinstance(msg, AIMessage) and msg.content:
                                msg_dict = message_to_dict(msg)
                                content_text = msg_dict["content"][0]["text"]
                                full_response += content_text
                                
                                # Send as complete message
                                msg_payload = {
                                    "values": [{
                                        "messages": [msg_dict],
                                        "checkpoint_id": f"checkpoint-{thread_id}-{current_checkpoint + 1}"
                                    }]
                                }
                                validate_sse_payload(msg_payload)
                                yield f"data: {json.dumps(msg_payload)}\n\n"
                except Exception as invoke_error:
                    print(f"Invoke fallback failed: {invoke_error}")
                    # Send error as response
                    error_msg = f"Agent error: {str(invoke_error)}"
                    full_response = error_msg
                    error_payload = {
                        "values": [{
                            "messages": [{
                                "role": "assistant", 
                                "content": [{"type": "text", "text": error_msg}]
                            }],
                            "checkpoint_id": f"checkpoint-{thread_id}-{current_checkpoint + 1}"
                        }]
                    }
                    validate_sse_payload(error_payload)
                    yield f"data: {json.dumps(error_payload)}\n\n"

            # Store assistant response in thread history
            if full_response:
                thread_data["messages"].append({"role": "assistant", "content": full_response})
                thread_data["checkpoint_id"] = f"checkpoint-{thread_id}-{len(thread_data['messages'])}"

            # Send completion event
            yield f"data: {json.dumps({'event': 'thread.run.completed', 'checkpoint_id': f'checkpoint-{thread_id}-{current_checkpoint + 1}'})}\n\n"

        except Exception as e:
            print(f"Stream error: {str(e)}")  # Debug logging
            error_msg = f"Streaming error: {str(e)}"
            error_payload = {
                "values": [{
                    "messages": [{
                        "role": "assistant", 
                        "content": [{"type": "text", "text": error_msg}]
                    }],
                    "checkpoint_id": f"checkpoint-{thread_id}-{current_checkpoint + 1}"
                }]
            }
            try:
                validate_sse_payload(error_payload)
                yield f"data: {json.dumps(error_payload)}\n\n"
            except:
                # Fallback error response
                yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
            
            yield f"data: {json.dumps({'event': 'thread.run.completed', 'checkpoint_id': f'checkpoint-{thread_id}-{current_checkpoint + 1}'})}\n\n"

    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ---------------- Run ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=2024, log_level="info")