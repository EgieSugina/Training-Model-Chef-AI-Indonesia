import os
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "./indonesian-food-model-final-Qwen-7B-500f-marge"
DATASET_PATH = "datasets/Indonesian_Food_Recipes_small.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(file_path):
    """
    Load and prepare the dataset for fine-tuning
    """
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(df)} recipes")
        
        # Prepare training data
        training_data = []
        
        for _, row in df.iterrows():
            # Extract recipe information
            title = row['Title']
            ingredients = row['Ingredients'].split('--')
            steps = row['Steps'].replace(')', '. ').replace('\n', ' ')
            
            # Create user prompt
            user_prompt = f"Bagaimana cara membuat {title}?"
            
            # Create assistant response
            assistant_response = f"Berikut resep untuk membuat {title}:\n\n"
            assistant_response += "Bahan-bahan:\n"
            for i, ingredient in enumerate(ingredients, 1):
                assistant_response += f"{i}. {ingredient.strip()}\n"
            
            assistant_response += "\nLangkah-langkah:\n"
            assistant_response += steps
            
            # Add to training data
            training_data.append({
                "user_prompt": user_prompt,
                "assistant_response": assistant_response
            })
        
        # Convert to Dataset
        dataset = Dataset.from_pandas(pd.DataFrame(training_data))
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def prepare_model():
    """
    Load and prepare the base model for fine-tuning
    """
    try:
        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        ) if DEVICE == "cuda" else None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, 
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # Prepare model for training
        if DEVICE == "cuda":
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        logger.info(f"Model prepared for fine-tuning on {DEVICE}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error preparing model: {e}")
        raise

def preprocess_function(examples, tokenizer):
    """
    Preprocess the dataset for training
    """
    system_message = (
        "Kamu adalah Chef Indonesia, asisten AI yang membantu membuat resep masakan Indonesia. "
        "Selalu berikan instruksi step by step yang jelas dan terperinci untuk setiap resep. "
        "Pastikan langkah-langkah diurutkan dengan baik dan mudah diikuti."
    )
    
    formatted_inputs = []
    
    for user_prompt, assistant_response in zip(examples["user_prompt"], examples["assistant_response"]):
        # Create chat format
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
        
        # Apply chat template
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        formatted_inputs.append(formatted_input)
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        formatted_inputs,
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    # Prepare labels (same as input_ids for causal language modeling)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

def train_model():
    """
    Fine-tune the model on the recipe dataset
    """
    try:
        # Load dataset
        dataset = load_dataset(DATASET_PATH)
        
        # Prepare model and tokenizer
        model, tokenizer = prepare_model()
        
        # Preprocess dataset
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True if DEVICE == "cuda" else False,
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )
        
        # Train model
        logger.info("Starting model fine-tuning...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        logger.info("Model fine-tuning completed successfully")
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    train_model()
