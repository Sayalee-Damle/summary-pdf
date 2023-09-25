from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path,PurePath

load_dotenv()

class Config:
    model = os.getenv("OPENAI_MODEL")
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        # request_timeout=request_timeout,
        # cache=has_langchain_cache,
        streaming=True,
    )
    path_embedding_dir = Path("./__pycache/vector_embedding.cpython-311.pyc")
    # used to create embeddings
    
    def apply_embedding_func(documents):
        emb_func = OpenAIEmbeddings()
        db = FAISS.from_documents(documents, emb_func)
        return db

cfg = Config()

if __name__ == "__main__":
    print("llm: ", cfg.llm)
    print("v embedding path: ", cfg.path_embedding_dir)