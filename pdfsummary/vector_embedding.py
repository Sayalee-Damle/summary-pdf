
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
import chromadb
from pathlib import Path
import user_input as ui

# if embedding dir is empty
def text_splitter(doc_to_split):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(doc_to_split)
    return documents

#put in config.py
def apply_embedding_func(documents):
    emb_func = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, emb_func)
    return db

#check if embedding dir is empty
def check_empty(collection):
    if collection.count() == 0:
        return True
    else:
        return False
    

#check if embedding dir exists else create
def init_vector_store():
    file_path = Path(__file__)
    emb_func = OpenAIEmbeddings()
    if file_path.exists() and len(list(file_path.glob("*"))) > 0:
        return FAISS.load_local(file_path.as_posix(), emb_func)
    else:
        path_pdf = ui.get_path_pdf()  
        loader = UnstructuredPDFLoader(path_pdf)
        pages = loader.load()
        documents = text_splitter(pages)
        vector_store = apply_embedding_func(documents)
        vector_store.save_local(file_path)
        return vector_store