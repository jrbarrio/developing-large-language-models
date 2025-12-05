# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

# Add formatting to the prompt
prompt="""
Instruction: Explain the concept of gravity in simple terms.
Question: What is gravity?
Answer:
"""

# Send the prompt to the model
output = llm(prompt, max_tokens=15, stop=["Question:"]) 
print(output['choices'][0]['text'])