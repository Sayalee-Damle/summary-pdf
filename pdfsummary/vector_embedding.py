
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS

import user_input as ui
from config import Config

# if embedding dir is empty
def text_splitter(doc_to_split):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(doc_to_split)
    db = Config.apply_embedding_func(documents)
    return db


#check if embedding dir is empty
def check_empty(collection):
    if collection.count() == 0:
        return True
    else:
        return False
    

#check if embedding dir exists else create
def init_vector_store():
    file_path = Config.path_embedding_dir
    emb_func = OpenAIEmbeddings()
    if file_path.exists() and len(list(file_path.glob("*"))) > 0:
        return FAISS.load_local(file_path.as_posix(), emb_func)
    else:
        return False
        path_pdf = ui.get_path_pdf()
        loader = UnstructuredPDFLoader(path_pdf)
        pages = loader.load()
        documents = text_splitter(pages)
        vector_store = apply_embedding_func(documents)
        vector_store.save_local(file_path)
        return vector_store