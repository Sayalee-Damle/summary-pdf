from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from pathlib import Path
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


# check if embedding dir exists else create
async def init_vector_store(path_pdf: Path) -> Tuple[FAISS, List[Document]]:
    # print("in init")
    file_path = (
        cfg.path_embedding_dir / path_pdf.stem
    )  # create a folder below path_embedding_dir
    documents_path = cfg.path_embedding_dir / f"{path_pdf.stem}_document"
    print(documents_path)
    if file_path.exists() and len(list(file_path.glob("*"))) > 0:
        # print("in if")
        db = FAISS.load_local(file_path.as_posix(), cfg.emb_func)
        with open(documents_path, "rb") as f:
            documents = pickle.load(f)
            return db, documents
    else:
        # print("in else")
        pages = extract_pages(path_pdf)
        # print("Got pages")
        db = await create_vector_store(pages)
        with open(documents_path, "wb") as f:
            pickle.dump(pages, f)
        if not file_path.exists():
            file_path.mkdir(parents=True)
        db.save_local(file_path.as_posix())
        return db, pages


def extract_pages(path_pdf: Path) -> List[Document]:
    loader = UnstructuredPDFLoader(path_pdf.as_posix())
    pages = loader.load()
    return pages


def convert_to_file(path_pdf: Path, target_path: Path):
    docs = extract_pages(path_pdf)
    with open(target_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.page_content)
            f.write("\n")


def convert_to_text(pages: List[Document]) -> str:
    return "\n".join([page.page_content for page in pages])


if __name__ == "__main__":
    import asyncio

    path_pdf = Path("C:/Users/Sayalee/Documents/langchain_research_paper.pdf")
    convert_to_file(path_pdf, cfg.save_pdf_here / "langchain_research_paper.txt")
    faiss, documents = asyncio.run(init_vector_store(path_pdf))
    assert faiss is not None
    assert documents is not None
    assert len(documents) > 0
    print(convert_to_text(documents))