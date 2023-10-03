from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path

load_dotenv()

class Config:
    model = os.getenv("OPENAI_MODEL")
    llm_cache = os.getenv("LLM_CACHE")  == 'True'
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        # request_timeout=request_timeout,
        cache=llm_cache,
        streaming=True,
    )
    path_embedding_dir = Path(os.getenv("EMBEDDING_DIR"))
    # used to create embeddings
    
    if not path_embedding_dir.exists():
        path_embedding_dir.mkdir(exist_ok=True, parents=True)

    emb_func = OpenAIEmbeddings()
    split_size = int(os.getenv("CHUNK_SIZE"))
    save_pdf_here = Path(os.getenv("PDF_PATH_DISC"))
    if not save_pdf_here.exists():
        save_pdf_here.mkdir(exist_ok=True, parents=True)
    ui_timeout = int(os.getenv("UI_TIMEOUT"))


cfg = Config()

if __name__ == "__main__":
    print("llm: ", cfg.llm)
    print("v embedding path: ", cfg.path_embedding_dir)