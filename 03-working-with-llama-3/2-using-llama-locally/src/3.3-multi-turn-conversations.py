chatbot = Conversation(llm, system_prompt="You are a travel expert that recommends a travel destination based on a prompt. Return the location name only as 'City, Country'.")

# Ask for the initial travel recommendation
first_recommendation = chatbot.create_completion("Recommend a Spanish-speaking city.")
print(first_recommendation)

# Add an additional request to update the recommendation
second_recommendation = chatbot.create_completion("A different city in the same country")
print(second_recommendation)