# Description
User interactive PDF reader. 
You will get a summary and keywords out of the PDF that you put in.
After you get the summary, you will get a chance to ask questions related to the topic and get a relevant answers for it.
Vector Embedding methods are used

# Packages used
1. openai
2. langchain
3. python-dotenv
4. FAISS
5. tiktoken
6. chainlit
7. unstructured

# Create project
## Installations prerequisites
1. install conda
2. pip install python-dotenv
3. pip install openai
4. pip install langchain
5. pip install tiktoken
6. pip install faiss-cpu
7. pip install prompt-toolkit
8. pip install chainlit
9. pip install unstructured

## Steps to create project
1. conda create -n convert_list python=3.11
2. conda activate convert_list
3. pip install poetry

### Create a pyproject.toml file
1. poetry init 
2. poetry install     