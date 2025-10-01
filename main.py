from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from graph import app

api = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str

# Endpoint principal para chat, bem básico do jeito que o FastAPI funciona
# Recebe a mensagem e o user_id (para contexto de usuário)
# Retorna a resposta final do grafo
@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    inputs = {"messages": [HumanMessage(content=request.message)], "user_id": request.user_id}
    
    final_answer = ""
    # Streaming a resposta do grafo
    # O grafo retorna chunks com o estado atual
    # No final pegamos a resposta final do AIMessage que não tem tool_calls
    for chunk in app.stream(inputs):
        print(f"--- Graph: {chunk} ---")
        
        # O chunk é um dict com o estado atual do grafo
        # Precisamos iterar para achar a resposta final
        for node_name, node_output in chunk.items():
            # Lista de mensagens no estado atual
            messages = node_output.get("messages", [])
            for message in messages:
                # A resposta final está em um AIMessage que tem conteúdo e NÃO tem tool_calls
                if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                    final_answer = message.content
    
    return {"response": final_answer}