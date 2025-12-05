class Conversation:
    # Complete the __init__ method of the Conversation class
    def __init__(self, llm: Llama, system_prompt='', history=[]):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}] + history

    def create_completion(self, user_prompt=''):
        # Add the user prompt to the history
        self.history.append({"role": "user", "content": user_prompt})
        # Send the history messages to the LLM
        output = self.llm.create_chat_completion(messages=self.history)
        conversation_result = output['choices'][0]['message']
        # Append the conversation_result to the history
        self.history.append(conversation_result)
        return conversation_result['content']