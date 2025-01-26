import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

urls = [
  "https://www.scotusblog.com/2025/01/supreme-court-upholds-tiktok-ban/",
  "http://www.scotusblog.com/?p=318475",
  "http://www.scotusblog.com/?p=318383",
  "https://en.wikipedia.org/wiki/TikTok_v._Garland"
]

api_key = os.getenv("OPENAI_API_KEY")

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

# Create embeddings for documents and store them in a vector store

vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(openai_api_key=api_key),
    persist_path="vectorstore",
    serializer="parquet")

vectorstore.persist()
