# build_vectorstore.py (Corrected)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv

load_dotenv()

# Esse codigo aqui so é para construir o vectorstore no Milvus
# Ele serve apenas para sair pegando oque for possivel dos sites da InfinitePay
# e jogar no Milvus para ser usado depois no RAG

# "DOCUMENTOS"
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

# WebLoader para pegar o conteudo dos sites
loader = WebBaseLoader(
    web_paths=urls,
    requests_kwargs={"headers": {"User-Agent": "cloudwalk-rag-builder/1.0"}},
)
docs = loader.load()

# --- ADDED FOR VERIFICATION ---
print(f"Conseguimos carregar tudo! {len(docs)} documentos dos sites.")
if docs:
    print("--- Snippet do primeiro documento: ---")
    print(docs[0].page_content[:500])
    print("------------------------------------")
    print("Caso queira verificar o DB, acesse localhost:7071 no seu navegador! e o Milvus Address é milvus-standalone2:19530")
# -----------------------------

# Parte de Chunk, sem muita beleza ou rigor tecnico, so para quebrar o texto em pedaços menores
# e facilitar a busca depois. Como é apenas para uma demostração, não precisa ser perfeito.
# Se fosse um projeto serio, teria que pensar melhor nisso.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Adjusted chunk size for better performance
splits = text_splitter.split_documents(docs)

# Milvus Vectorstore, apenas conectando e jogando os dados la dentro
# O nome da collection/coleção é "infinite_pay_docs"
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    #connection_args={"host": "localhost","port": 19530}, #LOCAL MILVUS
    connection_args={"uri": "http://standalone:19530"}, # DOCKER LOUCO
    collection_name="infinite_pay_docs",
    drop_old=True,
)

print("Vectorstore CRIADO no Milvus!! Operação concluída com sucesso!")