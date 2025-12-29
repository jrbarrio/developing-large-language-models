from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer
from transformers import AutoTokenizer

# Initialize PPO Configuration
gpt2_config = PPOConfig(model_name="gpt2", learning_rate=1.2e-5)

# Load the model
gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(gpt2_config.model_name)
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_config.model_name)

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    model=gpt2_model,
    config=gpt2_config,
    dataset=dataset_cs,
    tokenizer=gpt2_tokenizer
)