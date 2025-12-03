output = llm(
		"Can I exchange an item I purchased?", 
  		# Set the temperature parameter to provide more varied responses
		temperature=0.8,
        max_tokens=15
) 

print(output['choices'][0]['text'])