import chainlit as cl
from chainlit.types import AskFileResponse
from pathlib import Path
from langchain.schema import Document

from summarykeywords.config import cfg
import summarykeywords.extractkeyword as extractk
import summarykeywords.vector_embedding as ve

def write_to_disc(file: AskFileResponse) -> Path:
    content = file.content
    path = Path(file.path)
    with open(cfg.save_pdf_here / f"{path.stem}", "wb") as f:
        f.write(content)
    return cfg.save_pdf_here / f"{path.stem}"


def is_yes(input_msg: str) -> bool:
    return input_msg in ("yes", "y", "Yes", "Y")


@cl.on_chat_start
async def start() -> cl.Message:
    path_pdf = await get_pdf_path()
    if path_pdf:
        db, pages = await ve.init_vector_store(path_pdf)
        summary = await process_summary(pages)
        await process_translation(summary)
        await process_user_questions(db)
        await cl.Message(content=f"Thank You!").send()
        return


async def get_pdf_path() -> Path:
    files = None
    path_pdf = None

    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_files=1,
        ).send()

    path_pdf = write_to_disc(files[0])
    if not path_pdf.exists():
        await cl.Message(content=f"File upload failed, restart the chat").send()
        return None
    return path_pdf


async def process_user_questions(db) -> cl.Message:
    choiceq = None
    while choiceq == None:
        choiceq = await cl.AskUserMessage(
            content=f"Do you have any questions about this? (Yes/No)",
            timeout=cfg.ui_timeout,
        ).send()

    if not is_yes(choiceq["content"]):
        await cl.Message(content=f"Thank You!").send()
        return
    while True:
        ques = None
        while ques == None:
            ques = await cl.AskUserMessage(content=f"What is the Question?").send()

        answer = await extractk.question_llm(ques["content"], db)
        await cl.Message(content=f"{answer}").send()
        c = None
        while c == None:
            c = await cl.AskUserMessage(
                content=f"Do you have more questions (Yes/No)", timeout=cfg.ui_timeout
            ).send()
        if not is_yes(c["content"]):
            break


async def process_translation(summary) -> cl.Message:
    ans = None
    lang = None
    while ans == None:
        ans = await cl.AskUserMessage(
            content=f"Do you want to translate the summary? (Yes/No)",
            timeout=cfg.ui_timeout,
        ).send()

    if is_yes(ans["content"]):
        while lang == None:
            lang = await cl.AskUserMessage(
                content=f"Which Language do you want to use to translate the summary?",
                timeout=cfg.ui_timeout,
            ).send()
        translation = await extractk.translation(summary, lang["content"])

        await cl.Message(
            content=f"The translation of the summary file is: {translation}"
        ).send()
    else:
        await cl.Message(content=f"okay").send()


async def process_summary(pages) -> Document:
    await cl.Message(content=f"The summary will be shown below").send()
    summary = await extractk.summary_llm(pages)
    await cl.Message(content=f"{summary}").send()
    return summary
