from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI

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


cfg = Config()


if __name__ == "__main__":
    print("llm: ", cfg.llm)