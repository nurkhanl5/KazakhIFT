from huggingface_hub import login
#from kaggle_secrets import UserSecretsClient
#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
'''secret_label = "HF Hub"
secret_value = UserSecretsClient().get_secret(secret_label)
login(token=secret_value)
'''
# Replace "your_huggingface_token" with your actual Hugging Face token
huggingface_token = "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"

# Log in to Hugging Face Hub
login(token=huggingface_token)


import logging

logging.basicConfig(level=logging.INFO)
logging.info("Starting the script")
# Add more logging statements throughout the script

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import pandas as pd
import re
import numpy as np
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
import evaluate

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

import logging

logging.basicConfig(level=logging.INFO)
logging.info("Starting the script")
# Add more logging statements throughout the script

from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import transformers

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Set pad_token as end-of-sentence token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

import pandas as pd

df = pd.read_csv('/home/nurkhan.laiyk/nlp-project/merged_train_set.csv')
# Calculate the maximum length of text in the 'instruction' column
max_length_instruction = df['instruction'].apply(len).mean()

# Calculate the maximum length of text in the 'output' column
max_length_output = df['output'].apply(len).mean()

# Print the results
print(f"Maximum length of 'instruction': {max_length_instruction}")
print(f"Maximum length of 'output': {max_length_output}")
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

from datasets import Dataset

# Assuming `tokenized_df` is your pandas DataFrame
dataset = Dataset.from_pandas(tokenized_df[:10000])
tokenized_datasets = dataset.map(tokenize_function)# batched=True, # batch_size=...
tokenized_datasets = tokenized_datasets.remove_columns(['instruction', 'input','output'])


# Load LoRA configuration
peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora = True,
    target_modules=["q_proj", "k_proj", "o_proj", "up_proj", "gate_proj"],  # Target modules

)

# Set training parameters
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=2,
    # per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
#     evaluation_strategy="epoch",
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
#     max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)
from peft import get_peft_model, TaskType

peft_model = get_peft_model(model, 
                            peft_args)
print(print_number_of_trainable_model_parameters(peft_model))
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
#     eval_dataset=test_dataset,
    peft_config=peft_args,
    dataset_text_field="text",
#     max_seq_length=256,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained("./llama-3.1-8B-Kazakh-2")
tokenizer.save_pretrained("./llama-3.1-8B-Kazakh-2")