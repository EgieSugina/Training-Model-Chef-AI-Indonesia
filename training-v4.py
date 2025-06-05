import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset
import os
import json
import random
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# BASE_MODEL = "Qwen2.5-1.5B-Instruct"
# BASE_MODEL = "Qwen2.5-0.5B-Instruct"
BASE_MODEL = "Qwen2.5-3B-Instruct"
MODEL_NAME = f"Qwen/{BASE_MODEL}"
OUTPUT_DIR = f"./IFMF-{BASE_MODEL}-v4-small"
CSV_FILE = "./dataset_all/Indonesian_Food_Recipes_small.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model and tokenizer from {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Changed from "auto" to explicit float16
    device_map="auto",
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad token if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare the model for training with LoRA
print("Preparing model for training with LoRA...")
model = prepare_model_for_kbit_training(model)
base_model_config = model.config.to_dict()
if "model_type" not in base_model_config:
    base_model_config["model_type"] = "qwen"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the updated config
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(base_model_config, f)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM 
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # Added to enable gradient computation
print("LoRA adapters added to the model")

# Load and prepare dataset
print("Loading dataset...")
dataset_path = CSV_FILE

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please ensure the file exists.")

df = pd.read_csv(dataset_path)
print("Dataset columns:")
print(df.columns)
print(f"Total data in dataset: {len(df)}")

def clean_ingredients(ingredients_text):
    """Clean and extract ingredients from the text"""
    if pd.isna(ingredients_text):
        return []
    
    # Split by '--' and clean each ingredient
    ingredients = ingredients_text.split('--')
    cleaned = []
    for ing in ingredients:
        ing = ing.strip()
        if ing:
            # Remove numbers and measurements, keep only the ingredient name
            words = ing.split()
            # Find the main ingredient (usually the last meaningful word)
            main_ingredient = ' '.join([w for w in words if not any(char.isdigit() for char in w) and 
                                     w.lower() not in ['sdm', 'sdt', 'gram', 'kg', 'liter', 'ml', 'buah', 'lembar', 'batang', 'ruas', 'butir', 'ekor', 'ikat', 'gelas']])
            if main_ingredient:
                cleaned.append(main_ingredient.lower())
    return cleaned

def generate_training_examples(row):
    """Generate multiple training examples from one recipe"""
    examples = []
    title = row['Title']
    ingredients = row['Ingredients']
    steps = row['Steps']
    
    # Clean ingredients for recommendation feature
    ingredient_list = clean_ingredients(ingredients)
    
    system_msg = "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional. Tugasmu adalah memberikan resep lengkap, merekomendasikan masakan berdasarkan bahan yang tersedia, dan menjawab pertanyaan tentang masakan Indonesia."
    
    # Example 1: Standard recipe request
    user_msg1 = f"Tolong ajarkan saya resep lengkap untuk membuat {title}. Saya ingin mengetahui bahan-bahan dan langkah-langkahnya secara detail."
    assistant_msg1 = f"""Saya akan membantu Anda membuat {title}.

Berikut adalah bahan-bahan yang diperlukan:
{ingredients}

Langkah-langkah pembuatan:
{steps}

Tips:
- Pastikan semua bahan sudah disiapkan sebelum mulai memasak
- Ikuti langkah-langkah dengan teliti untuk hasil terbaik
- Sesuaikan tingkat kepedasan dan rasa sesuai selera"""
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg1},
            {"role": "assistant", "content": assistant_msg1}
        ]
    })
    
    # Example 2: Simple recipe question
    simple_questions = [
        f"Apa resep dari masakan {title}?",
        f"Bagaimana cara membuat {title}?",
        f"Bisa kasih resep {title}?",
        f"Resep {title} dong!",
        f"Cara masak {title} gimana?"
    ]
    
    user_msg2 = random.choice(simple_questions)
    assistant_msg2 = f"""Berikut resep {title}:

Bahan-bahan:
{ingredients}

Cara membuat:
{steps}

Selamat mencoba!"""
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg2},
            {"role": "assistant", "content": assistant_msg2}
        ]
    })
    
    # Example 3: Ingredient-based recommendation
    if len(ingredient_list) >= 3:
        # Select 3-5 random ingredients from the recipe
        sample_ingredients = random.sample(ingredient_list, min(random.randint(3, 5), len(ingredient_list)))
        ingredients_str = ', '.join(sample_ingredients)
        
        user_msg3 = f"Saya punya bahan: {ingredients_str}. Makanan apa yang bisa saya buat?"
        assistant_msg3 = f"""Dengan bahan yang Anda miliki ({ingredients_str}), saya merekomendasikan untuk membuat {title}!

Bahan lengkap yang diperlukan:
{ingredients}

Langkah pembuatan:
{steps}

Bahan yang sudah Anda miliki sudah cukup untuk membuat masakan ini. Tinggal lengkapi bahan lainnya jika diperlukan."""
        
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg3},
                {"role": "assistant", "content": assistant_msg3}
            ]
        })
    
    # Example 4: Ingredient inquiry
    if ingredient_list:
        main_ingredient = random.choice(ingredient_list)
        user_msg4 = f"Saya ada {main_ingredient}, bisa buat masakan apa?"
        assistant_msg4 = f"""Dengan {main_ingredient}, Anda bisa membuat {title}!

Ini resep lengkapnya:

Bahan-bahan:
{ingredients}

Cara membuat:
{steps}

{main_ingredient} adalah bahan utama dalam masakan ini."""
        
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg4},
                {"role": "assistant", "content": assistant_msg4}
            ]
        })
    
    return examples

def prepare_data(row):
    """Prepare training data from a recipe row"""
    training_examples = generate_training_examples(row)
    formatted_examples = []
    
    for example in training_examples:
        formatted_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        formatted_examples.append({"text": formatted_text})
    
    return formatted_examples

# Generate training examples
print("Generating training examples...")
all_examples = []
for _, row in df.iterrows():
    examples = prepare_data(row)
    all_examples.extend(examples)

print(f"Total training examples generated: {len(all_examples)}")

# Create dataset
dataset = Dataset.from_list(all_examples)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
print(f"Total examples after tokenization: {len(tokenized_dataset)}")

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,
    gradient_checkpointing=True,
    label_smoothing_factor=0.1,
    warmup_steps=100,
    eval_strategy="no",
    save_strategy="steps",
    load_best_model_at_end=False,
    report_to=None,  # Disable wandb logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    raise

# Save the model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training metadata
metadata = {
    "model_name": MODEL_NAME,
    "total_recipes": len(df),
    "total_training_examples": len(all_examples),
    "features": [
        "Recipe instruction",
        "Simple recipe questions",
        "Ingredient-based recommendations",
        "Ingredient inquiry responses"
    ]
}

with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Training completed and model saved!")
print(f"Model saved to: {OUTPUT_DIR}")
print(f"Total recipes processed: {len(df)}")
print(f"Total training examples: {len(all_examples)}")
print("\nModel capabilities:")
print("1. Memberikan resep lengkap masakan Indonesia")
print("2. Menjawab pertanyaan sederhana tentang resep")
print("3. Merekomendasikan masakan berdasarkan bahan yang tersedia")
print("4. Menjawab pertanyaan tentang penggunaan bahan tertentu")