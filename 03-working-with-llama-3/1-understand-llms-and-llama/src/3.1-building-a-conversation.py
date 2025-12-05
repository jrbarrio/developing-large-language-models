# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

prompt = "Give me four short steps to troubleshoot my internet connection."

conv = [
	# Complete the user message
	{
        "role": "user",
	    "content": prompt
    }
]

# Pass the conversation to the model
result = llm.create_chat_completion(messages=conv, max_tokens=20)
print(result)