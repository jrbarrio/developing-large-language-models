# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

question = "What is the most used database for data storage?"
response = llm(question)

# Extract the first choice and generated text
extracted = response["choices"][0]["text"]

# Print the extracted text
print(extracted)