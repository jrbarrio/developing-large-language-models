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