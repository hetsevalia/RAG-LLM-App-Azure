import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

dotenv.load_dotenv()
os.environ["USER_AGENT"] = "rag_llm_app"
DB_DOCS_LIMIT = 10

# Stream LLM response
def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})

# Load documents into DB
def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        # Pick loader by extension, not only MIME type
                        if doc_file.name.lower().endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.lower().endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.name.lower().endswith((".txt", ".md")):
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"⚠️ Document type {doc_file.type} not supported.")
                            continue

                        # Try to load
                        loaded_docs = loader.load()
                        if not loaded_docs:
                            st.error(f"❌ No content extracted from {doc_file.name}")
                            continue

                        docs.extend(loaded_docs)
                        st.session_state.rag_sources.append(doc_file.name)
                        st.success(f"✅ {doc_file.name} loaded")

                    except Exception as e:
                        st.error(f"❌ Error loading {doc_file.name}: {e}")
                        print(f"[DEBUG] Error loading {doc_file.name}: {e}")

                    finally:
                        # Clean up file
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                else:
                    st.error(f"❌ Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"📚 Document(s) loaded successfully into DB.", icon="✅")
        else:
            st.warning("⚠️ No documents were loaded.")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading URL {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL loaded successfully.", icon="✅")

            else:
                st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

def initialize_vector_db(docs):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # Keep last 20 collections in memory
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names.pop(0))

    return vector_db

def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Generate a search query for retrieval based on recent messages."),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
        You will answer user's queries based on context or your own knowledge.\n{context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({
        "messages": messages[:-1],
        "input": messages[-1].content
    }):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})
