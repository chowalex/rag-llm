import os

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

api_key = os.getenv("OPENAI_API_KEY")

vectorstore_path = "vectorstores/scotus_2023"
vectorstore = SKLearnVectorStore(
    persist_path=vectorstore_path,
    embedding=OpenAIEmbeddings(openai_api_key=api_key),
    serializer="parquet")

retriever = vectorstore.as_retriever(k=4)

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser() 

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)

# Example usage
question = "What did SCOTUS decide in nvidia v e. ohman, and can you briefly summarize the case?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)

question = "What did SCOTUS decide in Andrew v White?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)