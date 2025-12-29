from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
model = AutoModelForSequenceClassification.from_pretrained('openai-gpt')
config = RewardConfig(output_dir='output_dir', max_length=60)

# Initialize the reward trainer
trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
)