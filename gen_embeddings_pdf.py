import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from PyPDF2 import PdfReader

class PDFParser:
  def __init__(self, path):
    self.path = path

  def parse(self):
    contents = []
    with open(self.path,'rb') as f:
      pdf = PdfReader(f)
      for page in pdf.pages:
        contents.append(page.extract_text())
    return " ".join(contents)


pdfs = [
  "pdf/tiktok.pdf"
]

api_key = os.getenv("OPENAI_API_KEY")

docs = [PDFParser(pdf).parse() for pdf in pdfs]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = []
for doc in docs:
  doc_splits.extend(text_splitter.split_text(doc))

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

from langchain.schema import Document
doc_objects=[]
for i, chunk in enumerate(doc_splits):
  doc_objects.append(Document(page_content=chunk,metadata={"source":"scotus"}))

# Create embeddings for documents and store them in a vector store

vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_objects,
    embedding=OpenAIEmbeddings(openai_api_key=api_key),
    persist_path="vectorstore_pdf",
    serializer="parquet")

vectorstore.persist()
