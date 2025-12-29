from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig

# Load the pre-trained GPT-1 model for text classification
model = AutoModelForSequenceClassification.from_pretrained("openai-gpt")

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

# Initialize the reward configuration and set max_length
config = RewardConfig(output_dir="output_dir", max_length=60)