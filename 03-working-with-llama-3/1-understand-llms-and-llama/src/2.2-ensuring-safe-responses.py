# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

output = llm(
		"What are the symptoms of strep throat?", 
  		# Set the model parameters 
      	max_tokens=10, #Limit response length
		top_k=2 #Restrict word choices
) 

print(output['choices'][0]['text'])