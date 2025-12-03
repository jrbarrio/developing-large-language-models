output = llm(
      	"Write a tweet announcing a new analytics dashboard feature for enterprise users.", 
		max_tokens=15,
		# Set top-p to a value in the upper range for more varied responses
		top_p=0.9
	) 

print(output['choices'][0]['text'])