# Chef AI - Indonesian Food Recipe Assistant

Dokumentasi ini menjelaskan sistem Chef AI yang terdiri dari dua komponen utama: script pelatihan model (`training-v2.py`) dan script deployment model (`serve.py`). Sistem ini didesain untuk menyediakan asisten AI berbahasa Indonesia yang membantu pengguna dengan resep-resep masakan Indonesia.

## Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Script Pelatihan Model (training-v2.py)](#script-pelatihan-model-training-v2py)
  - [Konfigurasi Umum](#konfigurasi-umum)
  - [Fungsi Utama](#fungsi-utama)
  - [Alur Pelatihan](#alur-pelatihan)
  - [Teknik Optimasi](#teknik-optimasi)
- [Script Deployment (serve.py)](#script-deployment-servepy)
  - [Inisialisasi dan Setup](#inisialisasi-dan-setup)
  - [Fungsi Prediksi](#fungsi-prediksi)
  - [Antarmuka Gradio](#antarmuka-gradio)
  - [Deployment](#deployment)
- [Persyaratan Sistem](#persyaratan-sistem)

## Gambaran Umum

Chef AI adalah asisten berbasis AI untuk masakan Indonesia yang:
1. Dilatih pada dataset resep masakan Indonesia
2. Menggunakan model bahasa Transformer yang dioptimalkan
3. Diakses melalui antarmuka web yang dibangun dengan Gradio
4. Memberikan instruksi step-by-step untuk resep kuliner Indonesia

Sistem ini menggunakan model dasar berbahasa Indonesia (Nusantara-1.8B-Indo-Chat untuk serving, Qwen2-7B-Instruct untuk pelatihan) yang dilatih khusus untuk domain kuliner Indonesia.

## Script Pelatihan Model (training-v2.py)

### Konfigurasi Umum

Script ini menggunakan konfigurasi berikut:

```python
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "./IFMF-Qwen-7B-500food"
DATASET_FOLDER = "./datasets"
```

- `MODEL_NAME`: Model dasar yang digunakan (Qwen2 7B Instruct)
- `OUTPUT_DIR`: Direktori untuk menyimpan model terlatih
- `DATASET_FOLDER`: Direktori yang berisi dataset CSV resep

Serta pesan sistem untuk menentukan kepribadian AI:

```python
SYSTEM_MESSAGE = "Kamu adalah Chef Indonesia, asisten AI yang membantu membuat resep masakan Indonesia..."
```

### Fungsi Utama

#### 1. `load_datasets_from_folder(folder_path, max_recipes=5000)`

Fungsi ini bertanggung jawab untuk memuat dan memproses dataset resep.

**Langkah-langkah:**
- Mencari semua file CSV dalam folder yang ditentukan
- Memuat file CSV dalam chunk untuk efisiensi memori
- Membersihkan data (menghapus baris dengan nilai yang hilang)
- Memformat resep ke format yang sesuai untuk pelatihan model
- Menggabungkan semua dataset menjadi satu dataset tunggal
- Membatasi jumlah resep sesuai parameter `max_recipes`

**Optimasi memori:**
- Penggunaan `chunksize` untuk pembacaan bertahap
- Pembersihan memori dengan `del chunk` dan `gc.collect()`
- Penggunaan `dropna()` untuk menghilangkan data yang tidak lengkap

#### 2. `prepare_model_and_tokenizer()`

Fungsi ini menyiapkan model dan tokenizer dengan optimasi untuk GPU dengan memori terbatas.

**Langkah-langkah:**
- Memuat tokenizer dari model yang ditentukan
- Mengkonfigurasi quantization untuk mengurangi kebutuhan memori
- Memuat model dengan konfigurasi quantization
- Menerapkan LoRA (Low-Rank Adaptation) untuk fine-tuning efisien
- Menyiapkan model untuk pelatihan 4-bit

**Teknik Optimasi:**
- 4-bit quantization dengan BitsAndBytes
- Tensor dalam format float16
- Parameter LoRA yang dioptimalkan (r=8, lora_alpha=16)
- Target modul yang dipilih untuk efisiensi ("q_proj", "v_proj")

#### 3. `tokenize_dataset(dataset, tokenizer, max_length=512)`

Fungsi ini melakukan tokenisasi pada dataset untuk persiapan pelatihan.

**Fitur:**
- Truncation untuk membatasi panjang input
- Padding untuk menyeragamkan panjang sequence
- Pemrosesan batch untuk efisiensi
- Penghapusan kolom asli setelah tokenisasi

### Alur Pelatihan

Fungsi `main()` menjalankan alur pelatihan lengkap:

1. **Persiapan** - Membersihkan cache GPU
2. **Pemuatan Dataset** - Memanggil `load_datasets_from_folder`
3. **Split Dataset** - Membagi dataset menjadi training dan evaluasi (90:10)
4. **Inisialisasi Model** - Memanggil `prepare_model_and_tokenizer`
5. **Tokenisasi** - Memanggil `tokenize_dataset` untuk train dan test set
6. **Konfigurasi Pelatihan** - Menyiapkan TrainingArguments dengan parameter optimal
7. **Pelatihan** - Melatih model dengan early stopping
8. **Penyimpanan Model** - Menyimpan model, konfigurasi, dan tokenizer

### Teknik Optimasi

Script ini berisi berbagai teknik optimasi untuk GPU dengan memori terbatas (khususnya RTX 3060):

- **Manajemen Memori**:
  - Garbage collection (`gc.collect()`)
  - Pembersihan cache CUDA (`torch.cuda.empty_cache()`)
  
- **Quantization**:
  - 4-bit precision dengan BitsAndBytes
  - Double quantization
  - Compute dtype float16
  
- **Efisiensi Pelatihan**:
  - Batch size kecil (1) dengan gradient accumulation (4)
  - Mixed precision training (fp16)
  - Gradient clipping (max_grad_norm=0.3)
  - Early stopping (patience=3)
  
- **LoRA Fine-tuning**:
  - Low-rank adaptation untuk parameter-efficient fine-tuning
  - Target modul spesifik untuk mengurangi parameter yang dilatih

## Script Deployment (serve.py)

### Inisialisasi dan Setup

Script ini dimulai dengan inisialisasi dasar:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Kemudian memuat model dan tokenizer:

```python
model = AutoModelForCausalLM.from_pretrained(
    "kalisai/Nusantara-1.8B-Indo-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("kalisai/Nusantara-1.8B-Indo-Chat")
```

**Fitur Utama:**
- Deteksi otomatis device (GPU/CPU)
- Penerapan dtype otomatis sesuai device
- Device mapping otomatis untuk model besar
- Penanganan pad_token untuk generasi teks

### Fungsi Prediksi

Fungsi `predict(message, history)` adalah inti dari kemampuan generasi teks AI:

```python
def predict(message, history):
    # Format conversation history
    messages = [{"role": "system", "content": system_message}]
    # ...
```

**Langkah-langkah:**
1. Memformat pesan sistem dan riwayat percakapan
2. Menerapkan template chat ke format yang dimengerti model
3. Tokenisasi input dengan penambahan ke device yang sesuai
4. Menghasilkan token respons dengan model.generate()
5. Ekstraksi token baru (hanya respons, bukan input)
6. Decoding token menjadi teks

**Parameter Generasi:**
- `max_new_tokens=512`: Maksimum token yang dihasilkan
- `attention_mask`: Memastikan perhatian hanya pada token valid
- `pad_token_id`: Menggunakan padding token yang benar

### Antarmuka Gradio

Aplikasi web dibangun menggunakan Gradio dengan antarmuka chat:

```python
demo = gr.ChatInterface(
    predict,
    title="Nusantara AI Assistant",
    description="Asisten virtual berbahasa Indonesia",
    theme="soft"
)
```

**Fitur:**
- Antarmuka chat yang intuitif
- Tema "soft" untuk UX yang nyaman
- Integrasi dengan fungsi prediksi

### Deployment

Deployment aplikasi dilakukan dengan:

```python
# Add a health check endpoint
@demo.app.get("/health")
def health_check():
    return {"status": "healthy"}

# Launch the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Gradio server on port {port}...")
    demo.launch(server_port=port)
```

**Fitur Deployment:**
- Health check endpoint (`/health`)
- Konfigurasi port dinamis dari variabel lingkungan
- Server Gradio yang dapat diakses melalui web

## Persyaratan Sistem

### Dependensi Python
- transformers (v4.37.2 direkomendasikan)
- datasets (v2.16.1 direkomendasikan)
- peft (v0.7.1 direkomendasikan)
- torch
- gradio
- pandas
- accelerate
- bitsandbytes

### Hardware
- Training: GPU dengan VRAM minimal 12GB (optimized untuk RTX 3060)
- Inference: GPU atau CPU (inference lebih cepat dengan GPU)

### Perintah Instalasi
```bash
pip install transformers datasets torch accelerate peft bitsandbytes pandas gradio
# Atau dengan versi spesifik
pip install -U "transformers==4.37.2" "datasets==2.16.1" "peft==0.7.1"
``` 