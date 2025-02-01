import os
from pdf_parser import PDFParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class PDFVectorizer:
  def __init__(self, url, metadata = {}):
    self.url = url
    self.metadata = metadata

  def vectorize(
    self,
    vectorstore,
    chunk_size = 2560, 
    chunk_overlap = 256
  ):
    parser = PDFParser(self.url)
    contents = parser.parse()
    metadata = parser.metadata()

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    doc_splits = []
    doc_splits.extend(text_splitter.split_text(contents))

    # Create corresponding langchain documents.
    doc_objects = []
    for i, chunk in enumerate(doc_splits):
      doc_objects.append(
        Document(
          page_content = chunk,
          metadata = self.metadata
          )
      )

    vectorstore.add_documents(doc_objects)

    return vectorstore