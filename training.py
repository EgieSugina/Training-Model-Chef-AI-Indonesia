import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset
import os

# Load model and tokenizer
model_id = "kalisai/Nusantara-1.8b-Indo-Chat"
device = "cuda"#if torch.cuda.is_available() else "cpu"

print(f"Loading model and tokenizer from {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map=None,  # Removed device_map since we can't use it without Accelerate
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set pad token if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load and prepare dataset
print("Loading dataset...")
dataset_path = os.path.join("dataset", "Indonesian_Food_Recipes.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please ensure the file exists in the 'dataset' folder.")

df = pd.read_csv(dataset_path)

# Print dataset columns
print("Dataset columns:")
print(df.columns)

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

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./indonesian-food-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,
    prediction_loss_only=True,  # Only return loss during training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,  # Use the data collator instead of tokenizer
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
trainer.save_model("./indonesian-food-model-final")
print("Training completed!")
