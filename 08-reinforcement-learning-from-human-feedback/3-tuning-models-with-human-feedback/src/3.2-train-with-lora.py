from transformers import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

# Set the configuration parameters
config = LoraConfig(
    r=32,  
    lora_alpha=32,  
    lora_dropout=0.1,  
    bias="lora_only")  

# Apply the LoRA configuration to the 8-bit model
lora_model = get_peft_model(pretrained_model_8bit, config)
# Set up the tokenizer and model with a value head for PPO training
model = AutoModelForCausalLMWithValueHead.from_pretrained(lora_model)