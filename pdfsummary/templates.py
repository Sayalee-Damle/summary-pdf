template = """You display keywords and a summary and translate to {language}.
A user will pass in a pdf file, you should generate 5 keywords and a summary from it. It should be labelled 'Keywords: ' and 'Summary: ' on two different lines.
The summary will contain an appropriate title then please translate to {language}.
"""

system_template = """You will be a chatbot"""

template2 = """ You will give a relevent answer from the {output} for the {question} in the format:
Question : "what is the color of an apple?"
Answer : "Red"

Question : "who started Tesla?"
Answer : "Elon Musk"

"""