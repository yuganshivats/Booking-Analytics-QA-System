import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Clean the text file if it has been updated
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', '\n', text)
    return text

# Update the vector store if the text file has been updated
def update_vector_store():
    with open('./data/final_hotel_bookings_report.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    cleaned_text = clean_text(text)
    os.makedirs('./data/cleaned', exist_ok=True)
    with open('./data/cleaned/final_hotel_bookings_report.txt', 'w', encoding='utf-8') as file:
        file.write(cleaned_text)
    print("Vector store updated with cleaned report.")

def rag_chain():
    # Load the cleaned document
    text_loader = TextLoader('./data/cleaned/final_hotel_bookings_report.txt')
    documents = text_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory='chromadb')
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )

    rag_prompt = ChatPromptTemplate.from_template("""
    You are 'Lula', a QA system and hotel booking assistant.
    {context}
    Question: {question}
    Note: Do not include any extra information or disclaimer in the answer.
    """)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

if __name__ == "__main__":
    # Update the vector store only if the cleaned report does not exist
    update_vector_store()
    
    print("RAG chain created successfully.")
