from langchain.schema import BaseOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.vectorstores import FAISS

import summarykeywords.vector_embedding as ve
import summarykeywords.templates as t
from summarykeywords.config import cfg


class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")


def prompt_factory(system_template, human_template) -> ChatPromptTemplate:
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        template=system_template
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        template=human_template
    )
    messages = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    return chat_prompt


def get_db_pages(path_pdf) -> (FAISS, List[Document]):
    db, pages = asyncio.run(ve.init_vector_store(path_pdf))
    return db, pages


# to find summary
async def summary_llm(pages) -> Document:
    text = ve.convert_to_text(pages)
    chat_prompt = prompt_factory(t.summary_system_template, t.summary_human_template)
    chain = LLMChain(llm=cfg.llm, prompt=chat_prompt)
    summary = await chain.arun({"text": text})
    return summary


async def translation(summary, language) -> Document:
    chat_prompt_3 = prompt_factory(t.system_translate_template, t.translate_template)
    chain3 = LLMChain(llm=cfg.llm, prompt=chat_prompt_3)
    translate = await chain3.arun({"summary": summary, "language": language})
    return translate


# template for question answer
async def question_llm(query, db):
    chat_prompt_2 = prompt_factory(t.system_template, t.template2)
    docs = db.similarity_search(query)
    ve.convert_to_text(docs)
    chain2 = LLMChain(llm=cfg.llm, prompt=chat_prompt_2)
    ans = await chain2.arun({"output": docs, "question": query})
    return ans


if __name__ == "__main__":
    import asyncio
    path_pdf = Path("C:/Users/Sayalee/Documents/langchain_research_paper.pdf")
    db, pages = asyncio.run(ve.init_vector_store(path_pdf))
    print(asyncio.run(summary_llm(pages)))
    # print(pages)
