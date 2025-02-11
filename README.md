# RAG LLM Demo

This is a simple demo of a Retrieval-Augmented Generation (RAG) application using LangChain and Ollama for querying a vectorstore of embeddings from collections of PDF files. 
The implementation uses OpenAI embeddings with an SKLearn-based vector store to retrieve relevant documents and answer questions concisely using a language model.

## Installation

### Prerequisites
Ensure you have the following installed:

* Python 3.8+
* LangChain
* LangChain OpenAI
* LangChain Ollama
* scikit-learn
* OpenAI API key (set as OPENAI_API_KEY environment variable)

### Install Dependencies
```
pip install langchain langchain-openai langchain-ollama scikit-learn pandas
```

## Usage

### Setting Up the Environment

Ensure your OpenAI API key is set in the environment:
```
export OPENAI_API_KEY="your_openai_api_key"
```

### Running the Demo
Generate embeddings from PDFs linked on a web page, using `page_pdfs_vectorizer.py`.

Or, use existing datasets generated from Supreme Court opinions [here](https://huggingface.co/datasets/chowalex/scotus_opinions).

### Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

### License

This project is licensed under the MIT License.
