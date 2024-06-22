DB_PATH = "vectordb"
DATA_PATH = "data"

import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import time


def main():

    start_time = time.process_time()

    # Clear the existing database
    clear_database()

    # Create (or update) the vector database.
    documents = load_documents()
    chunks = split_documents(documents)
    ingest_documents(chunks)

    print(f"Process completed in {time.process_time() - start_time}s")


def load_documents():
    
    # Load data from a directory containing the pdf files
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def ingest_documents(chunks):

    # Load Vector DB
    vectordb = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory=DB_PATH)
    if len(chunks):
        print(f"Adding {len(chunks)} new documents in the DB...")
        vectordb.persist()
    else:
        print("No documents to add.")


def clear_database():

    if os.path.exists(DB_PATH):
        print("Clearing database...")
        shutil.rmtree(DB_PATH)
        print("Database cleared.")


if __name__ == "__main__":
    main()