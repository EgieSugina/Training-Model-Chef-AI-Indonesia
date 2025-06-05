"""
Fine-tuning script for Qwen models on Indonesian Food Recipe dataset

Citation:
If you find Qwen models helpful, please cite:

@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset
import os
import json
import random
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.utils.tensorboard import SummaryWriter

BASE_MODEL = "Qwen2.5-1.5B-Instruct"
# BASE_MODEL = "Qwen2.5-0.5B-Instruct"
# BASE_MODEL = "Qwen2.5-3B-Instruct"
MODEL_NAME = f"Qwen/{BASE_MODEL}"
OUTPUT_DIR = f"./IFMF-{BASE_MODEL}-v4-full"
CSV_FILE = "./dataset_all/Indonesian_Food_Recipes_full.csv"

# Initialize TensorBoard writer
tb_writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tensorboard"))

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model and tokenizer from {MODEL_NAME}...")

# Check if there's a checkpoint to resume from
checkpoint_dir = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_dir = os.path.join(OUTPUT_DIR, latest_checkpoint)
        print(f"Found checkpoint at {checkpoint_dir}. Resuming training...")

# Load model from checkpoint if exists, otherwise from base model
if checkpoint_dir:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
else:
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
    
    # Example 1: Standard recipe request variations
    recipe_request_variations = [
        f"Tolong ajarkan saya resep lengkap untuk membuat {title}. Saya ingin mengetahui bahan-bahan dan langkah-langkahnya secara detail.",
        f"Saya ingin belajar memasak {title}, bisa tolong jelaskan resepnya?",
        f"Bagaimana cara membuat {title} yang enak dan autentik?",
        f"Mau masak {title} untuk keluarga, bisa minta resep lengkapnya?",
        f"Tolong share resep {title} dong, termasuk tips-tips memasaknya!"
    ]
    
    for user_msg1 in recipe_request_variations:
        assistant_msg1 = f"""Saya akan membantu Anda membuat {title}.

Berikut adalah bahan-bahan yang diperlukan:
{ingredients}

Langkah-langkah pembuatan:
{steps}

Tips:
- Pastikan semua bahan sudah disiapkan sebelum mulai memasak
- Ikuti langkah-langkah dengan teliti untuk hasil terbaik
- Sesuaikan tingkat kepedasan dan rasa sesuai selera
- Gunakan api sedang untuk hasil masakan yang sempurna
- Jangan lupa cicipi sebelum disajikan"""
    
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg1},
                {"role": "assistant", "content": assistant_msg1}
            ]
        })
    
    # Example 2: Simple recipe questions with more variations
    simple_questions = [
        f"Apa resep dari masakan {title}?",
        f"Bagaimana cara membuat {title}?",
        f"Bisa kasih resep {title}?",
        f"Resep {title} dong!",
        f"Cara masak {title} gimana?",
        f"Minta resep {title} yang enak dong",
        f"Caranya bikin {title} apa aja ya?",
        f"Tutorial masak {title} dong",
        f"Bagi resep {title} yang autentik dong",
        f"Gimana sih cara bikin {title}?",
        f"Pengen masak {title}, resepnya apa aja?",
        f"Tolong ajarin bikin {title}",
        f"Butuh resep {title} nih",
        f"Bisa share resep {title}?",
        f"Mau coba masak {title}, resepnya gimana?"
    ]
    
    for user_msg2 in simple_questions:
        assistant_msg2 = f"""Berikut resep {title}:

Bahan-bahan:
{ingredients}

Cara membuat:
{steps}

Tips memasak:
- Pilih bahan-bahan yang segar
- Perhatikan tingkat kematangan
- Sesuaikan bumbu dengan selera
- Sajikan selagi hangat

Selamat mencoba!"""
    
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg2},
                {"role": "assistant", "content": assistant_msg2}
            ]
        })
    
    # Example 3: Ingredient-based recommendation with variations
    if len(ingredient_list) >= 3:
        # Select 3-5 random ingredients from the recipe
        sample_ingredients = random.sample(ingredient_list, min(random.randint(3, 5), len(ingredient_list)))
        ingredients_str = ', '.join(sample_ingredients)
        
        ingredient_questions = [
            f"Saya punya bahan: {ingredients_str}. Makanan apa yang bisa saya buat?",
            f"Di dapur ada {ingredients_str}, enaknya dimasak jadi apa ya?",
            f"Bahan yang tersedia: {ingredients_str}. Ada saran masakan?",
            f"Bisa masak apa dengan bahan {ingredients_str}?",
            f"Punya {ingredients_str}, bisa jadi masakan apa ya?",
            f"Mau masak tapi cuma punya {ingredients_str}, ada ide?",
            f"Rekomendasi masakan dengan bahan {ingredients_str} dong",
            f"Ada {ingredients_str} di kulkas, enaknya dibikin apa ya?",
            f"Bingung mau masak apa dengan {ingredients_str}",
            f"Chef, ada saran masakan dari bahan {ingredients_str}?"
        ]
        
        for user_msg3 in ingredient_questions:
            assistant_msg3 = f"""Dengan bahan yang Anda miliki ({ingredients_str}), saya merekomendasikan untuk membuat {title}!

Bahan lengkap yang diperlukan:
{ingredients}

Langkah pembuatan:
{steps}

Bahan yang sudah Anda miliki sudah cukup untuk membuat masakan ini. Tinggal lengkapi bahan lainnya jika diperlukan.

Tips:
- Pastikan bahan yang ada masih segar
- Jika ada bahan yang kurang, bisa disesuaikan dengan yang tersedia
- Ikuti langkah-langkah dengan teliti"""
        
            examples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg3},
                    {"role": "assistant", "content": assistant_msg3}
                ]
            })
    
    # Example 4: Ingredient inquiry with variations
    if ingredient_list:
        main_ingredient = random.choice(ingredient_list)
        ingredient_inquiry_variations = [
            f"Saya ada {main_ingredient}, bisa buat masakan apa?",
            f"Masakan apa yang bisa dibuat dari {main_ingredient}?",
            f"Ada {main_ingredient} di rumah, enaknya dimasak jadi apa ya?",
            f"Rekomendasi masakan dengan {main_ingredient} dong",
            f"Bingung mau masak apa dengan {main_ingredient}",
            f"Punya {main_ingredient}, ada ide masakan?",
            f"Bisa masak apa dengan {main_ingredient}?",
            f"Menu masakan dengan {main_ingredient} apa aja ya?",
            f"Ada saran olahan {main_ingredient}?",
            f"Mau masak pakai {main_ingredient}, ada rekomendasi?"
        ]
        
        for user_msg4 in ingredient_inquiry_variations:
            assistant_msg4 = f"""Dengan {main_ingredient}, Anda bisa membuat {title}!

Ini resep lengkapnya:

Bahan-bahan:
{ingredients}

Cara membuat:
{steps}

{main_ingredient} adalah bahan utama dalam masakan ini.

Tips pengolahan {main_ingredient}:
- Pilih {main_ingredient} yang segar dan berkualitas
- Cuci bersih sebelum diolah
- Potong sesuai petunjuk resep
- Simpan sisa {main_ingredient} dengan benar"""
        
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
    report_to=["tensorboard"],  # Enable TensorBoard logging
    resume_from_checkpoint=checkpoint_dir if checkpoint_dir else None,  # Enable resuming from checkpoint
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
    trainer.train(resume_from_checkpoint=checkpoint_dir if checkpoint_dir else None)
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    raise

# Save the model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Close TensorBoard writer
tb_writer.close()

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
print("\nTensorBoard logs available at:", os.path.join(OUTPUT_DIR, "tensorboard"))