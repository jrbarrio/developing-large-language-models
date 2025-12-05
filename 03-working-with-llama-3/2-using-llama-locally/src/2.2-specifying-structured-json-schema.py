# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

messages = [{'role': 'system', 'content': 'You are a helpful assistant that answers questions about space. You return your results in a JSON format with the Question and Answer fields.'}, {'role': 'user', 'content': 'How old is the Milky Way Galaxy?'}]

output = llm.create_chat_completion(
    messages=messages,
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            # Set the properties of the JSON fields and their data types
            "properties": {"Question": {"type": "string"}, "Answer": {"type": "string"}}
        }
    }
)

print(output['choices'][0]['message']['content'])