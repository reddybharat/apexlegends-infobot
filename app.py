from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time
import streamlit as st

DB_PATH = "vectordb"

def main():

    # Streamlit UI
    st.title("Apex Legends InfoBot")
    st.subheader("will answer your question on Apex Legends")

    input_query = st.text_input("Type your question here")

    get_response(input_query)


def create_document_chain_retriever():
    # Load Database
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

    # Create Chat Prompt Template
    user_prompt = ChatPromptTemplate.from_template(
    """
    Elaborate and answer the following question based only on the provided context.
    Think step by step before providing an answer.
    Please do not go out of context to answer the question, if the answer is not present in the given context then you dont have to answer the question.
    <context> {context} </context>
    Question : {input}
    """
    )

    # Model init, we are using llama3
    model_llama = Ollama(model="llama3")

    # Chain stuff document chain
    doc_chain = create_stuff_documents_chain(model_llama, user_prompt)

    # Retriever
    retriever = vectordb.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 5}
                    )
    
    # Retrieval Chain
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    return retrieval_chain


def get_response(input_query):

    if input_query:
        start_time = time.process_time()
        response = create_document_chain_retriever().invoke({"input":input_query})
        st.write(f"Response Time : {time.process_time() - start_time}s")
        st.write(f"Answer : {response['answer']}")
        sources = []
        with st.expander("Source(s)"):
            for r in enumerate(response['context']):
                sources.append(f"{r[1].metadata['source']} -> page {r[1].metadata['page']}")
            for source in get_unique_sources(sources):
                st.write(source)


def get_unique_sources(sources):
    uniques = []
    for s in sources:
        if s not in uniques:
            uniques.append(s)
    return uniques

if __name__ == "__main__":
    main()