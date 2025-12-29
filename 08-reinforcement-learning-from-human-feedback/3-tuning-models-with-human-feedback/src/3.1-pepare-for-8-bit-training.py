from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLMWithValueHead
from peft import prepare_model_for_int8_training

model_name = "gpt2"  

# Load the model in 8-bit precision
pretrained_model = AutoModelForCausalLM.from_pretrained(
                                                       model_name, 
                                                       load_in_8bit=True
                                                      )

# Prepare the model for fine-tuning
pretrained_model_8bit = prepare_model_for_int8_training(pretrained_model)

# Load the model with a value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_8bit)