# graph.py (Improved)

from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator

from tools import infinite_pay_rag_tool, web_search_tool, get_user_account_status, check_transfer_ability, reset_user_password, contact_support_agent

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    next_agent: str

# --- EXPANDED TOOLS FOR SUPPORT AGENT ---
knowledge_tools = [infinite_pay_rag_tool, web_search_tool]
support_tools = [get_user_account_status, check_transfer_ability, reset_user_password, contact_support_agent]

tool_node = ToolNode(knowledge_tools + support_tools)

# --- LLM SETUP ---
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
knowledge_llm = llm.bind_tools(knowledge_tools)
support_llm = llm.bind_tools(support_tools)

# --- AGENT NODES ---
def agent_node(state: AgentState, agent_llm, agent_name: str):
    config = {"configurable": {"user_id": state["user_id"]}}
    result = agent_llm.invoke(state["messages"], config=config)
    return {"messages": [result], "next_agent": agent_name}

def knowledge_agent_node_f(state: AgentState):
    return agent_node(state, knowledge_llm, "knowledge_agent")

def support_agent_node_f(state: AgentState):
    return agent_node(state, support_llm, "support_agent")

# --- IMPROVED ROUTER LOGIC ---
def router_logic(state: AgentState) -> str:
    """Melhor roteamento baseado em palavras-chave de suporte"""
    initial_message = state['messages'][0].content.lower()
    
    support_keywords = [
        "minha conta", "login", "senha", "acesso", "bloqueado", 
        "transferência", "pagar", "pagamento", "limite", "saldo",
        "problema", "ajuda", "suporte", "não consigo", "não funciona",
        "esqueci", "recuperar", "ativar", "desbloquear"
    ]
    
    if any(keyword in initial_message for keyword in support_keywords):
        print(f"--- Routing to Support Agent (keywords detected) ---")
        return "support_agent"
    else:
        print("--- Routing to Knowledge Agent ---")
        return "knowledge_agent"

def after_agent_router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return END

def after_tools_router(state: AgentState) -> str:
    return state["next_agent"]

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("knowledge_agent", knowledge_agent_node_f)
workflow.add_node("support_agent", support_agent_node_f)
workflow.add_node("tool_node", tool_node)

workflow.add_conditional_edges(
    "__start__",
    router_logic,
    {
        "knowledge_agent": "knowledge_agent",
        "support_agent": "support_agent"
    }
)

workflow.add_conditional_edges(
    "knowledge_agent",
    after_agent_router,
    {"tool_node": "tool_node", "__end__": END}
)
workflow.add_conditional_edges(
    "support_agent",
    after_agent_router,
    {"tool_node": "tool_node", "__end__": END}
)

workflow.add_conditional_edges(
    "tool_node",
    after_tools_router,
    {
        "knowledge_agent": "knowledge_agent",
        "support_agent": "support_agent"
    }
)

app = workflow.compile()