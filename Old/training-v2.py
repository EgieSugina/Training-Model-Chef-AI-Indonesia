import os
import gc
import glob
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig
from transformers import EarlyStoppingCallback

# Configuration for RTX 3060
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "./IFMF-Qwen2.5-1.5B-Instruct"
DATASET_FOLDER = "./dataset_all"

# System Message
SYSTEM_MESSAGE = "Kamu adalah Chef Indonesia, asisten AI yang membantu membuat resep masakan Indonesia. Selalu berikan instruksi step by step yang jelas dan terperinci untuk setiap resep yang kamu bagikan. Pastikan langkah-langkah diurutkan dengan baik dan mudah diikuti oleh pengguna."

def load_datasets_from_folder(folder_path, max_recipes=5000):
    """
    Load CSV files with memory-efficient approach
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in the folder: {folder_path}")
    
    all_datasets = []
    total_recipes_loaded = 0
    
    for file_path in csv_files:
        # Read CSV file in chunks to manage memory
        df_chunks = pd.read_csv(file_path, chunksize=1000)
        
        for chunk in df_chunks:
            # Clean and prepare data
            chunk = chunk.dropna(subset=['Title', 'Ingredients', 'Steps'])
            
            def format_recipe(row):
                try:
                    return {
                        "text": f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
Tolong jelaskan cara membuat {row['Title']} dengan detail lengkap.<|im_end|>
<|im_start|>assistant
Nama Resep: {row['Title']}

Bahan-Bahan:
{row['Ingredients']}

Langkah Memasak:
{row['Steps']}<|im_end|>"""
                    }
                except Exception as e:
                    print(f"Error processing recipe: {e}")
                    return None
            
            # Filter and format recipes
            recipes = chunk.apply(format_recipe, axis=1).dropna().tolist()
            
            # Convert to Dataset
            dataset = Dataset.from_list(recipes)
            all_datasets.append(dataset)
            
            total_recipes_loaded += len(dataset)
            
            # Stop if max recipes reached
            if total_recipes_loaded >= max_recipes:
                print(f"Reached maximum recipes limit: {max_recipes}")
                break
        
        # Free up memory
        del chunk
        gc.collect()
        
        if total_recipes_loaded >= max_recipes:
            break
    
    # Combine datasets
    combined_dataset = concatenate_datasets(all_datasets)
    print(f"Total recipes loaded: {len(combined_dataset)}")
    
    return combined_dataset

def prepare_model_and_tokenizer():
    """
    Optimized model and tokenizer loading for limited GPU memory
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Quantization and memory-efficient loading
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map='auto',  # Automatic device placement
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    
    # LoRA configuration optimized for memory
    lora_config = LoraConfig(
        r=8,  # Reduced rank for memory efficiency
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare and apply LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def tokenize_dataset(dataset, tokenizer, max_length=512*2):
    """
    Memory-efficient tokenization
    """
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
    
    # Use batched processing with smaller batch size
    return dataset.map(tokenize_function, batched=True, batch_size=32, remove_columns=dataset.column_names)

def main():
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load datasets
    dataset = load_datasets_from_folder(DATASET_FOLDER)
    
    # Split dataset with stratification
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset['train'], tokenizer)
    tokenized_eval_dataset = tokenize_dataset(dataset['test'], tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Training arguments optimized for RTX 3060
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced for 12GB VRAM
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Simulate larger batch size
        warmup_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,  # Mixed precision training
        max_grad_norm=0.3,  # Gradient clipping
    )
    
    # Initialize Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    model_to_save = trainer.model
    
    # Save the base model configuration with model_type
    base_model_config = model.config.to_dict()
    if "model_type" not in base_model_config:
        base_model_config["model_type"] = "qwen"  # Set model_type for Qwen model
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the updated config
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        import json
        json.dump(base_model_config, f)
    
    # Save PEFT model
    model_to_save.save_pretrained(OUTPUT_DIR)
    
    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()

# Requirements:
# pip install transformers datasets torch accelerate peft bitsandbytes pandas
# pip install -U "transformers==4.37.2" "datasets==2.16.1" "peft==0.7.1"