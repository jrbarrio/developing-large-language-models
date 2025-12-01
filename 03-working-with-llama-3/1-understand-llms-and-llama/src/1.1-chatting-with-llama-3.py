# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path=llama_path)

question = "What is the most used database for data storage?"
response = llm(question)
print(response)