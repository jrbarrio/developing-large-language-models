# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

class Conversation:
    def __init__(self, llm: Llama, system_prompt='', history=[]):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}] + history  
    
    def create_completion(self, user_prompt=''):  
        self.history.append({"role": "user", "content": user_prompt}) # Append input  
        output = self.llm.create_chat_completion(messages=self.history)  
        conversation_result = output['choices'][0]['message']
        self.history.append(conversation_result) # Append output
        return conversation_result['content'] # Return output

chatbot = Conversation(llm, system_prompt="You are a travel expert that recommends a travel destination based on a prompt. Return the location name only as 'City, Country'.")

# Ask for the initial travel recommendation
first_recommendation = chatbot.create_completion("Recommend a Spanish-speaking city.")
print(first_recommendation)

# Add an additional request to update the recommendation
second_recommendation = chatbot.create_completion("A different city in the same country")
print(second_recommendation)