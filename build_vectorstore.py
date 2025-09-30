# build_vectorstore.py
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
    # ... add all other URLs here
]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("main-content", "article-body", "post-content")
        )
    ),
    requests_kwargs={"headers": {"User-Agent": "cloudwalk-rag-builder/1.0"}},
)
docs = loader.load()

# 2. Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Create embeddings and store in Milvus
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    connection_args={
        "host": "localhost", # or "milvus" if running from another docker container
        "port": 19530,
    },
    collection_name="infinite_pay_docs", # Give your collection a name
    drop_old=True, # Optional: drops the collection if it already exists
)

print("Vectorstore created in Milvus successfully!")