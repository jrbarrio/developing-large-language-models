instruction = "You are a travel expert that recommends a travel destination based on a specification. Return the location name only in City, Country form."

# Define a chatbot using the Conversation class
chatbot = Conversation(llm, system_prompt=instruction)

# Send a prompt to the model
result = chatbot.create_completion("I'd like to learn about the Aztecs.")
print(result)