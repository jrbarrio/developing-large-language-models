# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

output = llm(
		"Can I exchange an item I purchased?", 
  		# Set the temperature parameter to provide more varied responses
		temperature=0.8,
        max_tokens=15
) 

print(output['choices'][0]['text'])