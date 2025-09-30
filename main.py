# main.py (Corrected)

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# You would import your compiled 'app' from graph.py
from graph import app # Assuming you saved your graph as 'app'

api = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    inputs = {"messages": [HumanMessage(content=request.message)], "user_id": request.user_id}
    
    final_answer = ""
    # Stream the output and get the last AI message
    for chunk in app.stream(inputs):
        print(f"--- Graph Chunk: {chunk} ---")
        
        # The chunk is a dictionary with the node name as the key
        for node_name, node_output in chunk.items():
            # Get the list of messages from the node's output
            messages = node_output.get("messages", [])
            for message in messages:
                # The final answer is in an AIMessage that has content and NO tool_calls
                if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                    final_answer = message.content
    
    return {"response": final_answer}