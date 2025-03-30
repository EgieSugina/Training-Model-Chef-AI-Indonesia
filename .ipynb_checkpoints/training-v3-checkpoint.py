import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./IFMF-Qwen2.5-1.5B-Instruct-small"
CSV_FILE = "./dataset_all/Indonesian_Food_Recipes_small.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model and tokenizer from {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",  # Use auto device mapping for better memory management
    # low_cpu_mem_usage=True,
    # load_in_8bit=True,  # Load model in 8-bit precision to reduce memory usage
    use_cache=False  # Disable KV cache for compatibility with gradient checkpointing
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad token if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare the model for training with LoRA
print("Preparing model for training with LoRA...")
model = prepare_model_for_kbit_training(model)
base_model_config = model.config.to_dict()
if "model_type" not in base_model_config: # Fixed: Removed unexpected indentation
    base_model_config["model_type"] = "qwen"  # Set model_type for Qwen model

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the updated config
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    import json
    json.dump(base_model_config, f)
# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    # Change task_type to TaskType.CAUSAL_LM
    task_type=TaskType.CAUSAL_LM 
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print("LoRA adapters added to the model")

# Load and prepare dataset
print("Loading dataset...")
dataset_path = CSV_FILE

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please ensure the file exists in the 'dataset' folder.")

df = pd.read_csv(dataset_path)

# Print dataset columns
print("Dataset columns:")
print(df.columns)
print(f"Total data in dataset: {len(df)}")

def prepare_data(row):
    # Format the recipe data into a chat format
    system_msg = "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional. Tugasmu adalah memberikan resep lengkap dengan bahan-bahan dan langkah memasak yang detail dan mudah diikuti."
    user_msg = f"Tolong ajarkan saya resep lengkap untuk membuat {row['Title']}. Saya ingin mengetahui bahan-bahan dan langkah-langkahnya secara detail."
    assistant_msg = f"""Saya akan membantu Anda membuat {row['Title']}.

Berikut adalah bahan-bahan yang diperlukan:
{row['Ingredients']}

Langkah-langkah pembuatan:
{row['Steps']}

Tips:
- Pastikan semua bahan sudah disiapkan sebelum mulai memasak
- Ikuti langkah-langkah dengan teliti untuk hasil terbaik
- Sesuaikan tingkat kepedasan dan rasa sesuai selera"""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ]
    
    # Apply chat template
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(prepare_data)
print(f"Total examples after preparation: {len(dataset)}")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=384  # Reduced from 512 to save memory
    )

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
print(f"Total examples after tokenization: {len(tokenized_dataset)}")

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced batch size to save memory
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,
    # prediction_loss_only=False,  # Only return loss during training
    # gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    # Adding label_smoothing_factor to prevent overfitting and potential numerical issues
    label_smoothing_factor=0.1 
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,  # Use the data collator instead of tokenizer
    # label_names=["input_ids", "attention_mask"],  # Explicitly provide label_names for PeftModelForCausalLM
)

# Start training


print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
print("Training completed!")
# https://colab.research.google.com/drive/1DlCVSptj0IsYxC0FpuxBhPPdkHTkQGCs?usp=sharing