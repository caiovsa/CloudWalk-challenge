# build_vectorstore.py (Corrected)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv

load_dotenv()

# 1. Load the documents
urls = [
    "https://www.infinitepay.io",
    "https://www.infinitepay.io/maquininha",
    "https://www.infinitepay.io/maquininha-celular",
    "https://www.infinitepay.io/tap-to-pay",
    "https://www.infinitepay.io/pdv",
    "https://www.infinitepay.io/receba-na-hora",
    "https://www.infinitepay.io/gestao-de-cobranca-2",
    "https://www.infinitepay.io/gestao-de-cobranca",
    "https://www.infinitepay.io/link-de-pagamento",
    "https://www.infinitepay.io/loja-online",
    "https://www.infinitepay.io/boleto",
    "https://www.infinitepay.io/conta-digital",
    "https://www.infinitepay.io/conta-pj",
    "https://www.infinitepay.io/pix",
    "https://www.infinitepay.io/pix-parcelado",
    "https://www.infinitepay.io/emprestimo",
    "https://www.infinitepay.io/cartao",
    "https://www.infinitepay.io/rendimento"
]

# The FIX is here: We removed the `bs_kwargs` argument entirely.
loader = WebBaseLoader(
    web_paths=urls,
    requests_kwargs={"headers": {"User-Agent": "cloudwalk-rag-builder/1.0"}},
)
docs = loader.load()

# --- ADDED FOR VERIFICATION ---
print(f"✅ Successfully loaded {len(docs)} documents from the websites.")
if docs:
    print("--- Snippet from the first document: ---")
    print(docs[0].page_content[:500])
    print("------------------------------------")
# -----------------------------

# 2. Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Adjusted chunk size for better performance
splits = text_splitter.split_documents(docs)

# 3. Create embeddings and store in Milvus
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    connection_args={
        "host": "localhost",
        "port": 19530,
    },
    collection_name="infinite_pay_docs",
    drop_old=True,
)

print("✅ Vectorstore created in Milvus successfully!")