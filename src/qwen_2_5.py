from unsloth import FastLanguageModel
import torch

from transformers import TextStreamer
#from google.colab import userdata


import pandas as pd
from datasets import Dataset
from datasets import load_dataset
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = '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', # use one if using gated models like meta-llama/Llama-2-7b-hf
)


df = pd.read_csv('merged_train_set.csv')
df.head()

def tokenize_function(row):
    # Tokenize the conversations
    question = ' '.join(row["instruction"]) if isinstance(row["instruction"], list) else row["instruction"]

    row['input_ids'] = tokenizer(question, padding="max_length", truncation=True, max_length = 128, return_tensors="pt").input_ids[0]
    
    # Assuming "answer" column is already a string, no need for conversion
    row['labels'] = tokenizer(row["output"], padding="max_length", truncation=True, max_length = 256, return_tensors="pt").input_ids[0]
    
    return row


# Tokenize the DataFrame
tokenized_df = df.apply(tokenize_function, axis=1)

# Convert columns to list
tokenized_df['input_ids'] = tokenized_df['input_ids'].apply(lambda x: x.tolist())
tokenized_df['labels'] = tokenized_df['labels'].apply(lambda x: x.tolist())


# Assuming `tokenized_df` is your pandas DataFrame
dataset = Dataset.from_pandas(tokenized_df[:10000])
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request in Kazakh.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


dataset = dataset.map(formatting_prompts_func, batched = True,)

# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        logging_steps = 500,
        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        warmup_steps = 5,
        num_train_epochs = 4,

        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer_stats = trainer.train()
model.save_pretrained("qwen_results") # Local saving
tokenizer.save_pretrained("qwen_results")


model_name = "qwen_results"  # Replace with your model path if it's local
max_seq_length = 512
dtype = "float16"
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_training(model)  # Prepare the model for training/inference

# Read the test data
test_data = pd.read_csv('/home/nurkhan.laiyk/nlp-project/merged_test_set.csv')

# Ensure test_data contains 'instruction' and 'input' columns
if 'instruction' not in test_data.columns or 'input' not in test_data.columns:
    raise ValueError("Test data must contain 'instruction' and 'input' columns.")

# Define a function for generating responses
def generate_response(row):
    # Create the input prompt
    prompt = alpaca_prompt.format(row['instruction'], row['input'], "")
    
    # Tokenize and move to GPU
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Set up the streamer for text output
    text_streamer = TextStreamer(tokenizer)
    
    # Generate response
    outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=64)
    
    # Decode the response
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_response

# Apply the function to each row in the test data
test_data['Owen Responses'] = test_data.apply(generate_response, axis=1)

# Save the updated test data to a new CSV file
output_file = '/home/nurkhan.laiyk/nlp-project/test_with_owen_responses.csv'
test_data.to_csv(output_file, index=False)

print(f"Responses saved to {output_file}")