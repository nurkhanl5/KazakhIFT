import pandas as pd
from unsloth.models import FastLanguageModel
from transformers import TextStreamer

# Model configuration
model_name = "qwen_results"  # Replace with your model path if it's local
max_seq_length = 1024
dtype = "float16"
load_in_4bit = False

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Prepare the model for inference

# Read the test data
test_data = pd.read_csv('/home/nurkhan.laiyk/nlp-project/merged_test_set.csv')

# Ensure test_data contains 'instruction' and 'input' columns
if 'instruction' not in test_data.columns or 'input' not in test_data.columns:
    raise ValueError("Test data must contain 'instruction' and 'input' columns.")

# Sample 100 rows from the dataset
sampled_data = test_data.sample(n=10, random_state=42).reset_index(drop=True)

# Define Alpaca-style prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request in Kazakh.

### Instruction:
{}

### Input:
{}

### Response:
{} <|endoftext|>
"""
# Define a function for generating responses
def generate_response(row):
    # Create the input prompt
    prompt = alpaca_prompt.format(row['instruction'], row['input'], "")
    
    # Tokenize and move to GPU
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    
    # Set up the streamer for text output
    text_streamer = TextStreamer(tokenizer)
    
    # Generate response
    outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)
    
    # Decode the response
    generated_response = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    return generated_response

# Apply the function to each row in the sampled data
sampled_data['Owen Responses'] = sampled_data.apply(generate_response, axis=1)

# Save the updated sampled data to a new CSV file
output_file = '/home/nurkhan.laiyk/nlp-project/sampled_test_with_owen_responses_2.csv'
sampled_data.to_csv(output_file, index=False)

print(f"Responses for sampled data saved to {output_file}")
