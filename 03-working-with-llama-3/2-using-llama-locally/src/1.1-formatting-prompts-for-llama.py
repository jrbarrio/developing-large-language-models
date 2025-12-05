# Add formatting to the prompt
prompt="""
Instruction: Explain the concept of gravity in simple terms.
Question: What is gravity?
Answer:
"""

# Send the prompt to the model
output = llm(prompt, max_tokens=15, stop=["Question:"]) 
print(output['choices'][0]['text'])