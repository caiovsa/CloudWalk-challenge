# graph.py (Corrected Graph Structure)

from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator

# Import your tools from tools.py
from tools import infinite_pay_rag_tool, web_search_tool, get_user_account_status, check_transfer_ability

# --- 1. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    # This new field will track which agent is currently active.
    next_agent: str

# --- 2. Define the Tools and the ToolNode ---
tools = [infinite_pay_rag_tool, web_search_tool, get_user_account_status, check_transfer_ability]
tool_node = ToolNode(tools)

# --- 3. Define the LLM and the Agents ---
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
knowledge_llm = llm.bind_tools([infinite_pay_rag_tool, web_search_tool])
support_llm = llm.bind_tools([get_user_account_status, check_transfer_ability])

# --- 4. Define the Agent Nodes ---
def agent_node(state: AgentState, agent_llm, agent_name: str):
    config = {"configurable": {"user_id": state["user_id"]}}
    result = agent_llm.invoke(state["messages"], config=config)
    # We return an AIMessage and also update the 'next_agent' state field.
    return {"messages": [result], "next_agent": agent_name}

def knowledge_agent_node_f(state: AgentState):
    return agent_node(state, knowledge_llm, "knowledge_agent")

def support_agent_node_f(state: AgentState):
    return agent_node(state, support_llm, "support_agent")

# --- 5. Define the Router Logic ---
def router_logic(state: AgentState) -> str:
    """This function is now ONLY a conditional edge function."""
    initial_message = state['messages'][0].content.lower()
    if any(keyword in initial_message for keyword in ["sign in", "transfer", "my account", "can't"]):
        print("--- Routing to Support Agent ---")
        return "support_agent"
    else:
        print("--- Routing to Knowledge Agent ---")
        return "knowledge_agent"

def after_agent_router(state: AgentState) -> str:
    """Router that decides what to do after an agent has acted."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return END

def after_tools_router(state: AgentState) -> str:
    """After tools, route back to the agent that called them."""
    return state["next_agent"]

# --- 6. Build the Graph ---
workflow = StateGraph(AgentState)

# Add all nodes to the graph
workflow.add_node("knowledge_agent", knowledge_agent_node_f)
workflow.add_node("support_agent", support_agent_node_f)
workflow.add_node("tool_node", tool_node)

# THE FIX: The entry point is now a conditional edge that uses our router_logic.
# This is the correct way to start with a decision.
workflow.add_conditional_edges(
    "__start__", # Special keyword for the graph's entry point
    router_logic,
    {
        "knowledge_agent": "knowledge_agent",
        "support_agent": "support_agent"
    }
)

# Edges after an agent has run
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

# Edge after tools have been executed
workflow.add_conditional_edges(
    "tool_node",
    after_tools_router,
    {
        "knowledge_agent": "knowledge_agent",
        "support_agent": "support_agent"
    }
)

# Compile the final graph
app = workflow.compile()