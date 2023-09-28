
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader, PyPDFium2Loader
from PyPDF2 import PdfReader, PdfWriter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS
from pathlib import Path
import user_input as ui
from config import cfg
from typing import Tuple, List
from langchain.schema import Document
import pickle

# if embedding dir is empty
async def create_vector_store(doc_to_split) -> FAISS:
    print("in create_vector_store")
    text_splitter = CharacterTextSplitter(chunk_size=cfg.split_size, chunk_overlap=10)
    documents = text_splitter.split_documents(doc_to_split)
    db = FAISS.from_documents(documents, cfg.emb_func)
    return db


#check if embedding dir exists else create
async def init_vector_store(path_pdf: Path) -> Tuple[FAISS, List[Document]]:
    print("in init")
    file_path = cfg.path_embedding_dir/path_pdf.stem    #create a folder below path_embedding_dir
    documents_path = cfg.path_embedding_dir/f"{path_pdf.stem}_document"
    print(documents_path)
    if file_path.exists() and len(list(file_path.glob("*"))) > 0:
        print("in if")
        db = FAISS.load_local(file_path.as_posix(), cfg.emb_func)
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)
            return db, documents
    else:
        print("in else")
        loader = PyPDFium2Loader(str(path_pdf))
        pages = loader.load()
        print("Got pages")
        db = await create_vector_store(pages)
        with open(documents_path, 'wb') as f:
            pickle.dump(pages, f)
        if not file_path.exists():
            file_path.mkdir(parents = True)
        db.save_local(file_path.as_posix())
        return db, pages[0]

if __name__ == "__main__":
    path_pdf = Path("/Users/sayaleedamle/WorkDocumets/samplepdf3.pdf")
    faiss, documents =  init_vector_store(path_pdf) 
    assert faiss is not None
    assert documents is not None
    assert len(documents) > 0
    print(faiss)
    