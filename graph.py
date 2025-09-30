# graph.py (Final, Corrected Version)

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode # <-- THE NEW, CORRECT IMPORT
from typing import TypedDict, Annotated
import operator

# Import your tools from tools.py
from tools import infinite_pay_rag_tool, web_search_tool, get_user_account_status, check_transfer_ability

# --- 1. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str

# --- 2. Define the Tools and the ToolNode ---
tools = [infinite_pay_rag_tool, web_search_tool, get_user_account_status, check_transfer_ability]

# This single line replaces BOTH the ToolExecutor and the manual tool_node function.
tool_node = ToolNode(tools)

# --- 3. Define the LLM and the Agents ---
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Bind the tools to specific LLMs for each agent's purpose
knowledge_llm = llm.bind_tools([infinite_pay_rag_tool, web_search_tool])
support_llm = llm.bind_tools([get_user_account_status, check_transfer_ability])

# --- 4. Define the Agent Nodes ---
def agent_node(state: AgentState, agent_llm):
    """A generic node that invokes the specified agent LLM."""
    # We pass the user_id in the config so it's available to tools if needed
    config = {"configurable": {"user_id": state["user_id"]}}
    result = agent_llm.invoke(state["messages"], config=config)
    return {"messages": [result]}

def knowledge_agent_node_f(state: AgentState):
    """The node for the knowledge agent."""
    return agent_node(state, knowledge_llm)

def support_agent_node_f(state: AgentState):
    """The node for the customer support agent."""
    return agent_node(state, support_llm)

# --- 5. Define the Router Logic ---
def router_logic(state: AgentState):
    """This router decides the next step based on the last message."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "tool_node" # Route to the pre-built ToolNode
        
    initial_message = state['messages'][0].content.lower()
    if any(keyword in initial_message for keyword in ["sign in", "transfer", "my account", "can't"]):
        return "support_agent"

    # Route to the knowledge agent by default
    return "knowledge_agent"
    
def after_agent_router(state: AgentState):
    """Router to take after an agent has acted."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return END

# --- 6. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("knowledge_agent", knowledge_agent_node_f)
workflow.add_node("support_agent", support_agent_node_f)
workflow.add_node("tool_node", tool_node) # <-- Use the pre-built ToolNode directly

workflow.set_entry_point("knowledge_agent")

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

# After the tools are executed, route back to the knowledge_agent to process the results
workflow.add_edge("tool_node", "knowledge_agent")

# Compile the final graph
app = workflow.compile()