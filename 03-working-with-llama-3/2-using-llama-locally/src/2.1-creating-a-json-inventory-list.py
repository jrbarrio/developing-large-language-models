# Import the Llama class
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(model_path="03-working-with-llama-3/models/Meta-Llama-3-8B.Q8_0.gguf", verbose=False)

output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You convert inventory lists from text to JSON, extracting item counts and names from the text as keys and values in the form: item: count; for example, 'banana': 32.",},
            {"role": "user", "content": "Fifteen apples, thirty-three oranges, and five thousand fifty-two potatoes."},
        ],
		# Specify output format to JSON
        response_format={
            "type": "json_object",
        }
)

print(output['choices'][0]['message']['content'])