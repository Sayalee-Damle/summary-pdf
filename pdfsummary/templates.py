summary_template = """You display keywords and a summary from the {text} provided.
A user will pass in a pdf file, you should generate 5 keywords and a summary from it. It should be labelled 'Keywords: ' and 'Summary: ' on two different lines.
The summary will contain an appropriate title.
"""

system_translate_template = """You are a translator"""
translate_template ="""
You will translate {summary} to {language}
labelled as:  "Summary of the Content in {language}"
"""


system_template = """You will be a chatbot"""

template2 = """ You will give a relevent answer from the {output} for the {question} in the format:
Question : "what is the color of an apple?"
Answer : "Red"

Question : "who started Tesla?"
Answer : "Elon Musk"

"""