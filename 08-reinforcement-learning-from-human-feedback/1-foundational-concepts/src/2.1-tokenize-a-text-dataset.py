from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("argilla/tripadvisor-hotel-reviews")

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

# Add padding with the pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)