from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
model_name = "deepseek-ai/DeepSeek-R1"  # Changed to 32B model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote=True)

# Set the device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare input
input_text = "What is the meaning of life?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)