from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Ferramenta de web search genérica
# Usei o TavilySearch que é free e simples
# Pode ser trocado por qualquer outra ferramenta de web search
# OBS: Prefiro muito mais o perplexity...mas ai ele ja é pago
web_search_tool = TavilySearch(k=3)

# RAG Chain (Etapa basica de RAG)
def create_rag_chain():
    vectorstore = Milvus(
        embedding_function=OpenAIEmbeddings(),
        connection_args={"uri": "http://standalone:19530"},
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

rag_chain = create_rag_chain()

# Primeira ferramenta, para perguntas sobre produtos InfinitePay
@tool
def infinite_pay_rag_tool(query: str) -> str:
    """Use para perguntas sobre produtos InfinitePay"""
    print("--- Calling InfinitePay RAG Tool ---")
    response = rag_chain.invoke({"input": query})
    return response["answer"]


# Aqui começam as ferramentas específicas para o agente de suporte
# Essas ferramentas são mais focadas em ações que o agente pode fazer
# como verificar status de conta, resetar senha, etc.
# Tudo mockado, mas dá para ter uma ideia legal...
# Ser sincero isso aqui tinha como ser bem melhor, mas é só para demo mesmo acho que serve
@tool
def get_user_account_status(user_id: str) -> str:
    """Verifica status da conta do usuário. Use quando usuário perguntar sobre status da conta, login ou problemas de acesso."""
    print(f"--- Checking account status for user: {user_id} ---")
    
    # Mock data mais realista
    user_status_map = {
        "cliente123": {"status": "active", "level": "premium", "since": "2024-01-15"},
        "cliente456": {"status": "active", "level": "basic", "since": "2024-03-20"},
        "cliente789": {"status": "blocked", "level": "gold", "since": "2023-11-05", "reason": "Verificação pendente"},
        "usuario999": {"status": "inactive", "reason": "Conta suspensa por violação de termos"}
    }
    
    user_data = user_status_map.get(user_id, {"status": "not_found", "reason": "Usuário não encontrado"})
    
    if user_data["status"] == "active":
        return f"Conta ATIVA - Nível: {user_data['level']} - Cliente desde: {user_data['since']}"
    elif user_data["status"] == "blocked":
        return f"Conta BLOQUEADA - Motivo: {user_data['reason']} - Nível anterior: {user_data['level']}"
    elif user_data["status"] == "inactive":
        return f"Conta INATIVA - Motivo: {user_data['reason']}"
    else:
        return "Conta não encontrada no sistema"

@tool
def check_transfer_ability(user_id: str) -> str:
    """Verifica se usuário pode fazer transferências. Use quando perguntarem sobre transferências, pagamentos ou limites."""
    print(f"--- Checking transfer ability for user: {user_id} ---")
    
    # Mock data melhorado
    transfer_status_map = {
        "cliente123": {"can_transfer": True, "daily_limit": 5000.00, "available_today": 3500.00},
        "cliente456": {"can_transfer": True, "daily_limit": 2000.00, "available_today": 1500.00},
        "cliente789": {"can_transfer": False, "reason": "Verificação de segurança pendente"},
        "usuario999": {"can_transfer": False, "reason": "Conta suspensa"}
    }
    
    status = transfer_status_map.get(user_id, {"can_transfer": False, "reason": "Usuário não encontrado"})
    
    if status["can_transfer"]:
        return f"Transferências LIBERADAS - Limite diário: R$ {status['daily_limit']:.2f} - Disponível hoje: R$ {status['available_today']:.2f}"
    else:
        return f"Transferências BLOQUEADAS - Motivo: {status['reason']}"

@tool
def reset_user_password(user_id: str) -> str:
    """Solicita reset de senha para o usuário. Use quando usuário esquecer a senha."""
    print(f"--- Password reset for user: {user_id} ---")
    return f"Email de recuperação enviado para {user_id}@email.com. O link expira em 1 hora."

@tool
def contact_support_agent(user_id: str) -> str:
    """Conecta usuário com agente humano. Use para problemas complexos não resolvidos pelas ferramentas automáticas."""
    print(f"--- Connecting user {user_id} to human agent ---")
    return f"Agente humano solicitado para {user_id}. Tempo de espera estimado: 3 minutos. Caso ID: SUP-{user_id}-2024"