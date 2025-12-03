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