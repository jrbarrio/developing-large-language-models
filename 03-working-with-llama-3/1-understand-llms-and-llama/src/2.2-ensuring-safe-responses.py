output = llm(
		"What are the symptoms of strep throat?", 
  		# Set the model parameters 
      	max_tokens=10, #Limit response length
		top_k=2 #Restrict word choices
) 

print(output['choices'][0]['text'])