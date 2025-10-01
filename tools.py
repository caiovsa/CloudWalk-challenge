from langchain.tools import tool
# DEPRECATION FIX: Import the new Tavily Search
from langchain_tavily import TavilySearch
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()



# --- Fix the Deprecation Warning ---
# This line addresses the first warning in your traceback.
web_search_tool = TavilySearch(k=3)

# --- Define and create the RAG Chain (this part is mostly the same) ---
def create_rag_chain():
    vectorstore = Milvus(
        embedding_function=OpenAIEmbeddings(),
        #connection_args={"host": "milvus", "port": 19530}, # Local Milvus
        connection_args={"uri": "http://standalone:19530"}, # Docker louco
        collection_name="infinite_pay_docs"
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    system_prompt = (
        "Você é um assistente para tarefas de perguntas e respostas. "
        "Use as seguintes peças de contexto recuperado para responder à pergunta. "
        "Se você não souber a resposta, apenas diga que não sabe. "
        "Use no máximo três frases e mantenha a resposta concisa. "
        "\n\n"
        "CONTEXTO:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# --- THE MAIN FIX IS HERE ---
# 1. Create the chain once.
rag_chain = create_rag_chain()

# 2. Create a simple function decorated with @tool that CALLS the chain.
@tool
def infinite_pay_rag_tool(query: str) -> str:
    """
    Use this tool to answer questions about InfinitePay's products, services, fees,
    and any other information found on the infinitepay.io website.
    """
    print("--- Calling InfinitePay RAG Tool ---")
    response = rag_chain.invoke({"input": query})
    print(f"RAG Response: {response}")
    return response["answer"]

# --- Tools for Customer Support Agent ---
# These are mock tools. In a real scenario, they would query a database.

@tool
def get_user_account_status(user_id: str) -> dict:
    """Gets the account status for a given user ID. Returns mock data."""
    print(f"--- Checking account status for user: {user_id} ---")
    if "789" in user_id:
        return {"status": "active", "account_level": "gold", "has_pending_transfers": False}
    else:
        return {"status": "inactive", "reason": "Account not found"}

@tool
def check_transfer_ability(user_id: str) -> dict:
    """Checks if a user is able to make transfers and provides a reason if not. Returns mock data."""
    print(f"--- Checking transfer ability for user: {user_id} ---")
    if "789" in user_id:
        # Simulate a temporary block
        return {"can_transfer": False, "reason": "A security check is pending. Please check your email for verification steps."}
    else:
        return {"can_transfer": False, "reason": "Account not found."}