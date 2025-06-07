"""
🍳 PROGRAM PELATIHAN AI CHEF INDONESIA 🍳
====================================

Program ini untuk melatih AI agar bisa jadi chef Indonesia yang pintar!
AI ini bisa:
- Memberikan resep masakan Indonesia
- Merekomendasikan masakan dari bahan yang ada
- Menjawab pertanyaan tentang memasak

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

# 📚 BAGIAN 1: IMPORT LIBRARY
# ===========================
# Seperti memanggil teman-teman untuk membantu kita memasak
import torch  # 🔥 Library untuk deep learning (otak AI)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling  # 🤖 Tools untuk AI
import pandas as pd  # 📊 Untuk baca file Excel/CSV (data resep)
from datasets import Dataset  # 📝 Untuk mengatur data training
import os  # 📁 Untuk mengatur file dan folder
import json  # 📄 Untuk menyimpan pengaturan
import random  # 🎲 Untuk acak data
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType  # ⚡ Teknologi LoRA untuk training efisien
from torch.utils.tensorboard import SummaryWriter  # 📈 Untuk monitor progress training

# 🎯 BAGIAN 2: PENGATURAN DASAR
# ==============================
# Ini seperti mengatur resep utama yang akan kita buat

# Pilih model AI yang akan dilatih (seperti memilih chef mana yang akan kita ajari)
# BASE_MODEL = "Qwen2.5-1.5B-Instruct"  # Model sedang (dikomentari)
BASE_MODEL = "Qwen2.5-0.5B-Instruct"     # Model kecil (dipilih karena cepat)
# BASE_MODEL = "Qwen2.5-3B-Instruct"     # Model besar (dikomentari)

# Buat nama lengkap model
MODEL_NAME = f"Qwen/{BASE_MODEL}"

# Tentukan folder untuk menyimpan hasil training
OUTPUT_DIR = f"./IFMF-{BASE_MODEL}-v4-small"

# Lokasi file data resep masakan Indonesia
CSV_FILE = "./dataset_all/Indonesian_Food_Recipes_small.csv"

# 📊 Setup monitor untuk melihat progress training (seperti dashboard mobil)
tb_writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tensorboard"))

# 💻 Deteksi apakah komputer punya GPU (kartu grafis) untuk mempercepat training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Menggunakan device: {device}")

print(f"📥 Loading model dan tokenizer dari {MODEL_NAME}...")

# 🔄 BAGIAN 3: CEK CHECKPOINT (RESUME TRAINING)
# =============================================
# Seperti cek apakah kita pernah memasak setengah jadi sebelumnya

checkpoint_dir = None
if os.path.exists(OUTPUT_DIR):
    # Cari semua checkpoint yang pernah disimpan
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        # Ambil checkpoint yang paling baru (nomor paling besar)
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_dir = os.path.join(OUTPUT_DIR, latest_checkpoint)
        print(f"🔄 Ditemukan checkpoint di {checkpoint_dir}. Melanjutkan training...")

# 🤖 BAGIAN 4: LOAD MODEL DAN TOKENIZER
# =====================================
# Seperti memanggil chef AI dan mengajarinya bahasa kita

if checkpoint_dir:
    # Jika ada checkpoint, lanjutkan dari situ
    print("🔄 Memuat model dari checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,  # Gunakan format angka yang efisien
        device_map="auto",          # Otomatis atur GPU/CPU
        use_cache=False            # Matikan cache untuk training
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
else:
    # Jika belum pernah training, mulai dari model dasar
    print("🆕 Memuat model dasar...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Format angka efisien (setengah presisi)
        device_map="auto",          # Otomatis atur perangkat
        use_cache=False            # Matikan cache untuk training
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 🔧 Set token padding (seperti memberi tanda jeda dalam kalimat)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✅ Token padding telah diatur")

# ⚡ BAGIAN 5: SETUP LORA (TRAINING EFISIEN)
# ==========================================
# LoRA = Low-Rank Adaptation, seperti hanya mengubah bumbu tanpa ganti seluruh resep

print("⚡ Mempersiapkan model untuk training dengan LoRA...")
model = prepare_model_for_kbit_training(model)  # Siapkan model untuk training efisien

# Dapatkan konfigurasi model dan pastikan ada tipe model
base_model_config = model.config.to_dict()
if "model_type" not in base_model_config:
    base_model_config["model_type"] = "qwen"

# 📁 Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Folder output: {OUTPUT_DIR}")

# 💾 Simpan konfigurasi model
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(base_model_config, f)

# 🎯 Konfigurasi LoRA (Parameter Efficient Fine-tuning)
print("🎯 Mengatur konfigurasi LoRA...")
lora_config = LoraConfig(
    r=16,                     # 📐 Rank: Berapa banyak parameter baru (16 = balance bagus)
    lora_alpha=32,           # 🔢 Alpha: Kekuatan perubahan (32 = 2x dari rank)
    target_modules=[         # 🎯 Bagian model yang akan diubah:
        "q_proj", "v_proj", "k_proj", "o_proj",    # Self-attention (fokus AI)
        "gate_proj", "up_proj", "down_proj"        # Feed-forward (pemrosesan)
    ],
    lora_dropout=0.05,       # 📉 Dropout: Cegah overfitting (5%)
    bias="none",             # 🚫 Tidak ubah bias
    task_type=TaskType.CAUSAL_LM  # 📝 Tipe tugas: Generasi teks
)

# Terapkan LoRA ke model
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # Aktifkan gradien untuk input
print("✅ Adapter LoRA berhasil ditambahkan ke model")

# 📊 BAGIAN 6: LOAD DAN PERSIAPKAN DATASET
# =========================================
# Seperti menyiapkan bahan-bahan masakan dari buku resep

print("📊 Memuat dataset...")
dataset_path = CSV_FILE

# Cek apakah file dataset ada
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ File dataset tidak ditemukan di {dataset_path}. Pastikan file ada!")

# Baca file CSV berisi resep-resep masakan
df = pd.read_csv(dataset_path)
print("📋 Kolom dalam dataset:")
print(df.columns)
print(f"📊 Total data dalam dataset: {len(df)}")

def clean_ingredients(ingredients_text):
    """
    🧹 FUNGSI PEMBERSIH BAHAN MASAKAN
    
    Fungsi ini seperti seorang asisten dapur yang:
    - Membaca daftar bahan mentah
    - Memisahkan setiap bahan
    - Menghilangkan ukuran/takaran (2 sdm, 500 gram, dll)
    - Hanya ambil nama bahan utamanya saja
    
    Input: "2 sdm minyak goreng--500 gram daging sapi--3 buah bawang merah"
    Output: ["minyak goreng", "daging sapi", "bawang merah"]
    """
    # Jika tidak ada data, return list kosong
    if pd.isna(ingredients_text):
        return []
    
    # Pisahkan bahan berdasarkan tanda '--'
    ingredients = ingredients_text.split('--')
    cleaned = []
    
    for ing in ingredients:
        ing = ing.strip()  # Hapus spasi di awal/akhir
        if ing:
            # Pisahkan jadi kata-kata
            words = ing.split()
            # Ambil kata-kata yang bukan angka dan bukan satuan
            main_ingredient = ' '.join([w for w in words if not any(char.isdigit() for char in w) and 
                                     w.lower() not in ['sdm', 'sdt', 'gram', 'kg', 'liter', 'ml', 'buah', 'lembar', 'batang', 'ruas', 'butir', 'ekor', 'ikat', 'gelas']])
            if main_ingredient:
                cleaned.append(main_ingredient.lower())
    
    return cleaned

def generate_training_examples(row):
    """
    🔄 FUNGSI GENERATOR CONTOH TRAINING
    
    Fungsi ini seperti guru yang kreatif, dari 1 resep bisa bikin banyak variasi pertanyaan:
    
    Input: 1 resep rendang
    Output: 40+ variasi pertanyaan & jawaban tentang rendang:
    - "Bagaimana cara membuat rendang?"
    - "Resep rendang dong!"
    - "Saya punya daging sapi, bisa masak apa?"
    - "Cara mengolah daging sapi gimana?"
    - dll...
    
    Ini membuat AI jadi pintar menjawab berbagai macam gaya pertanyaan
    """
    examples = []
    title = row['Title']        # Nama masakan
    ingredients = row['Ingredients']  # Bahan-bahan
    steps = row['Steps']        # Langkah-langkah
    
    # Bersihkan daftar bahan untuk fitur rekomendasi
    ingredient_list = clean_ingredients(ingredients)
    
    # Pesan sistem: Memberitahu AI perannya sebagai Chef Indonesia
    system_msg = "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional. Tugasmu adalah memberikan resep lengkap, merekomendasikan masakan berdasarkan bahan yang tersedia, dan menjawab pertanyaan tentang masakan Indonesia."
    
    # 📝 TIPE 1: PERMINTAAN RESEP STANDAR (5 variasi)
    # Pertanyaan formal untuk minta resep lengkap
    recipe_request_variations = [
        f"Tolong ajarkan saya resep lengkap untuk membuat {title}. Saya ingin mengetahui bahan-bahan dan langkah-langkahnya secara detail.",
        f"Saya ingin belajar memasak {title}, bisa tolong jelaskan resepnya?",
        f"Bagaimana cara membuat {title} yang enak dan autentik?",
        f"Mau masak {title} untuk keluarga, bisa minta resep lengkapnya?",
        f"Tolong share resep {title} dong, termasuk tips-tips memasaknya!"
    ]
    
    # Buat jawaban untuk setiap variasi pertanyaan
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
    
        # Tambahkan ke list contoh training
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg1},
                {"role": "assistant", "content": assistant_msg1}
            ]
        })
    
    # ❓ TIPE 2: PERTANYAAN RESEP SEDERHANA (15 variasi)
    # Pertanyaan casual/santai untuk minta resep
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
    
    # Buat jawaban yang lebih ringkas untuk pertanyaan sederhana
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
    
    # 🥘 TIPE 3: REKOMENDASI BERDASARKAN BAHAN (10 variasi)
    # User punya beberapa bahan, minta saran masakan
    if len(ingredient_list) >= 3:
        # Pilih 3-5 bahan secara acak dari resep
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
    
    # 🔍 TIPE 4: PERTANYAAN TENTANG BAHAN SPESIFIK (10 variasi)
    # User punya 1 bahan tertentu, mau tahu bisa dimasak apa
    if ingredient_list:
        main_ingredient = random.choice(ingredient_list)  # Pilih 1 bahan secara acak
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
    """
    📝 FUNGSI PERSIAPAN DATA
    
    Fungsi ini seperti editor yang merapikan semua contoh training:
    - Ambil semua variasi dari 1 resep
    - Format jadi template chat yang benar
    - Siapkan untuk tokenisasi
    
    Input: 1 baris resep
    Output: 40+ contoh training yang sudah diformat
    """
    training_examples = generate_training_examples(row)
    formatted_examples = []
    
    # Format setiap contoh jadi template chat
    for example in training_examples:
        formatted_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        formatted_examples.append({"text": formatted_text})
    
    return formatted_examples

# 🔄 BAGIAN 7: GENERATE SEMUA CONTOH TRAINING
# ===========================================
# Proses semua resep jadi ribuan contoh training

print("🔄 Membuat contoh-contoh training...")
all_examples = []
for _, row in df.iterrows():  # Loop setiap baris resep
    examples = prepare_data(row)      # Buat variasi untuk resep ini
    all_examples.extend(examples)     # Tambahkan ke koleksi total

print(f"✅ Total contoh training yang dibuat: {len(all_examples)}")

# Buat dataset dari list contoh
dataset = Dataset.from_list(all_examples)

def tokenize_function(examples):
    """
    🎯 FUNGSI TOKENISASI
    
    Fungsi ini seperti penerjemah yang mengubah teks jadi angka:
    - AI tidak mengerti teks, hanya mengerti angka
    - Setiap kata diubah jadi ID angka
    - Padding = tambahkan angka 0 jika teks terlalu pendek
    - Truncation = potong jika teks terlalu panjang
    - Max length 512 = maksimal 512 kata per contoh
    """
    return tokenizer(
        examples["text"],
        padding="max_length",    # Samakan panjang semua teks
        truncation=True,         # Potong jika terlalu panjang
        max_length=512          # Maksimal 512 token
    )

# 🎯 Tokenisasi seluruh dataset
print("🎯 Tokenisasi dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,                    # Proses dalam batch untuk efisiensi
    remove_columns=dataset.column_names  # Hapus kolom teks asli, tinggal angka
)
print(f"✅ Total contoh setelah tokenisasi: {len(tokenized_dataset)}")

# 📦 BAGIAN 8: DATA COLLATOR
# ==========================
# Seperti tukang parkir yang merapikan data dalam batch

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # False = Causal LM (prediksi kata berikutnya), bukan Masked LM
)

# ⚙️ BAGIAN 9: KONFIGURASI TRAINING
# =================================
# Seperti mengatur resep training: berapa lama, seberapa cepat, dll

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                    # 📁 Folder output
    num_train_epochs=3,                       # 🔄 Berapa kali lihat seluruh data (3x)
    per_device_train_batch_size=2,           # 📦 Berapa contoh per batch (2)
    gradient_accumulation_steps=4,            # 🔁 Kumpulkan 4 batch sebelum update (total = 2x4=8)
    save_steps=500,                          # 💾 Simpan checkpoint setiap 500 langkah
    save_total_limit=2,                      # 🗂️ Maksimal simpan 2 checkpoint
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),  # 📋 Folder log
    logging_steps=100,                       # 📊 Log progress setiap 100 langkah
    learning_rate=2e-5,                      # 📈 Seberapa cepat belajar (0.00002)
    weight_decay=0.01,                       # ⚖️ Regularisasi untuk cegah overfitting
    fp16=True if torch.cuda.is_available() else False,  # 🔥 Gunakan FP16 jika ada GPU
    gradient_checkpointing=True,             # ✂️ Hemat memori dengan gradient checkpointing
    label_smoothing_factor=0.1,              # 🎯 Smoothing untuk regularisasi (10%)
    warmup_steps=100,                        # 🔥 Pemanasan 100 langkah pertama
    eval_strategy="no",                      # 🚫 Tidak evaluasi (hanya training)
    save_strategy="steps",                   # 💾 Simpan berdasarkan langkah
    load_best_model_at_end=False,           # 🚫 Tidak load model terbaik di akhir
    report_to=["tensorboard"],              # 📊 Report ke TensorBoard
    resume_from_checkpoint=checkpoint_dir if checkpoint_dir else None,  # 🔄 Resume dari checkpoint
)

# 🏋️ BAGIAN 10: INISIALISASI TRAINER
# ===================================
# Seperti menyiapkan pelatih AI yang akan mengajari model

trainer = Trainer(
    model=model,                    # 🤖 Model yang akan dilatih
    args=training_args,            # ⚙️ Pengaturan training
    train_dataset=tokenized_dataset,  # 📊 Data training
    data_collator=data_collator,   # 📦 Pengatur batch data
)

# 🚀 BAGIAN 11: MULAI TRAINING!
# =============================
# Ini bagian paling penting: AI mulai belajar!

print("🚀 Memulai training...")
print("⏱️ Proses ini bisa memakan waktu lama tergantung:")
print("   - Ukuran dataset")
print("   - Kecepatan komputer/GPU")
print("   - Jumlah epoch")
print("📊 Monitor progress di TensorBoard!")

try:
    # Mulai training dengan kemungkinan resume dari checkpoint
    trainer.train(resume_from_checkpoint=checkpoint_dir if checkpoint_dir else None)
    print("🎉 Training berhasil diselesaikan!")
except Exception as e:
    print(f"❌ Training gagal dengan error: {e}")
    raise

# 💾 BAGIAN 12: SIMPAN MODEL HASIL TRAINING
# =========================================
# Seperti menyimpan chef AI yang sudah pintar ke dalam kotak

print("💾 Menyimpan model...")
trainer.save_model(OUTPUT_DIR)      # Simpan model yang sudah ditraining
tokenizer.save_pretrained(OUTPUT_DIR)  # Simpan tokenizer

# 📈 Tutup TensorBoard writer
tb_writer.close()

# 🗂️ BAGIAN 13: SIMPAN METADATA TRAINING
# ======================================
# Seperti mencatat resep dan hasil masakan

metadata = {
    "model_name": MODEL_NAME,
    "total_recipes": len(df),
    "total_training_examples": len(all_examples),
    "features": [
        "Recipe instruction",                    # Instruksi resep
        "Simple recipe questions",              # Pertanyaan resep sederhana
        "Ingredient-based recommendations",     # Rekomendasi dari bahan
        "Ingredient inquiry responses"          # Respons pertanyaan bahan
    ]
}

with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# 🎉 BAGIAN 14: PESAN AKHIR & RINGKASAN
# =====================================
print("\n" + "="*50)
print("🎉 TRAINING SELESAI! 🎉")
print("="*50)
print(f"📁 Model disimpan di: {OUTPUT_DIR}")
print(f"📊 Total resep yang diproses: {len(df)}")
print(f"🔢 Total contoh training: {len(all_examples)}")
print(f"⚡ Menggunakan teknologi LoRA untuk efisiensi")
print(f"🎯 Model menggunakan: {BASE_MODEL}")

print("\n🤖 KEMAMPUAN AI CHEF INDONESIA:")
print("1. 🍽️ Memberikan resep lengkap masakan Indonesia")
print("2. ❓ Menjawab pertanyaan sederhana tentang resep")
print("3. 🥘 Merekomendasikan masakan berdasarkan bahan tersedia")
print("4. 🔍 Menjawab pertanyaan tentang penggunaan bahan tertentu")

print(f"\n📊 TensorBoard logs tersedia di: {os.path.join(OUTPUT_DIR, 'tensorboard')}")
print("💡 Gunakan: tensorboard --logdir [path] untuk melihat grafik training")
print("\n✅ Sekarang AI Chef Indonesia siap digunakan!")