from langchain.schema import BaseOutputParser
from config import Config
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
import vector_embedding as ve
import templates as t
from pathlib import Path


class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")


def prompt_factory(system_template, human_template):
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        template=system_template
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        template=human_template
    )
    messages = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    return chat_prompt


def get_db_pages(path_pdf):
    # path_pdf = user_input.get_path_pdf()
    db, pages = asyncio.run(ve.init_vector_store(path_pdf))
    return db, pages


# to find summary
async def summary_llm(pages):
    text = ve.convert_to_text(pages)
    chat_prompt = prompt_factory(t.summary_template, "{text}")
    chain = LLMChain(llm=Config.llm, prompt=chat_prompt)
    summary = await chain.arun({"text": text})
    return summary


async def translation(summary, language):
    chat_prompt_3 = prompt_factory(t.system_translate_template, t.translate_template)
    chain3 = LLMChain(llm=Config.llm, prompt=chat_prompt_3)
    translate = await chain3.arun({"summary": summary, "language": language})
    return translate


# print(pages[0].page_content)


# template for question answer
async def question_llm(query, db):
    chat_prompt_2 = prompt_factory(t.system_template, t.template2)
    docs = db.similarity_search(query)
    ve.convert_to_text(docs)
    chain2 = LLMChain(llm=Config.llm, prompt=chat_prompt_2)
    ans = await chain2.arun({"output": docs, "question": query})
    return ans


if __name__ == "__main__":
    import asyncio

    path_pdf = Path("C:/Users/Sayalee/Documents/langchain_research_paper.pdf")
    db, pages = asyncio.run(ve.init_vector_store(path_pdf))
    print(asyncio.run(summary_llm(pages)))
    # print(pages)
