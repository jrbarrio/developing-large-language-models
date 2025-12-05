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

instruction = "You are a travel expert that recommends a travel destination based on a specification. Return the location name only in City, Country form."

# Define a chatbot using the Conversation class
chatbot = Conversation(llm, system_prompt=instruction)

# Send a prompt to the model
result = chatbot.create_completion("I'd like to learn about the Aztecs.")
print(result)