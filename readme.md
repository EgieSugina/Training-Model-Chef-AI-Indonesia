# 🍳 AI Chef Indonesia - Indonesian Food Recipe AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-green.svg)](https://huggingface.co/transformers)
[![LoRA](https://img.shields.io/badge/LoRA-Efficient%20Fine--tuning-orange.svg)](https://github.com/microsoft/LoRA)

## 📖 Overview

AI Chef Indonesia adalah model AI yang dilatih khusus untuk menjadi chef virtual yang ahli dalam masakan tradisional Indonesia. Model ini menggunakan arsitektur Qwen2.5 dengan fine-tuning menggunakan teknologi LoRA untuk efisiensi.

### 🎯 Kemampuan AI Chef Indonesia

- 🍽️ **Resep Lengkap**: Memberikan resep masakan Indonesia dengan detail bahan dan langkah-langkah
- ❓ **Q&A Masakan**: Menjawab pertanyaan tentang teknik memasak dan bahan
- 🥘 **Rekomendasi Cerdas**: Merekomendasikan masakan berdasarkan bahan yang tersedia
- 🔍 **Panduan Bahan**: Memberikan tips pengolahan bahan tertentu

## 🏗️ Arsitektur Model

### Base Model
- **Model**: Qwen2.5-0.5B-Instruct
- **Parameter**: ~500M parameters
- **Architecture**: Transformer-based Causal Language Model
- **Context Length**: 512 tokens

### Graph Layers & Architecture Details

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Chef Indonesia                       │
├─────────────────────────────────────────────────────────────┤
│  Input Layer (Tokenization)                               │
│  ├── Tokenizer: Qwen2.5 Tokenizer                        │
│  ├── Max Length: 512 tokens                              │
│  └── Padding Strategy: max_length                        │
├─────────────────────────────────────────────────────────────┤
│  Embedding Layer                                          │
│  ├── Hidden Size: 1024                                   │
│  ├── Vocab Size: 151936                                  │
│  └── Position Embedding: Rotary Position Embedding       │
├─────────────────────────────────────────────────────────────┤
│  Transformer Layers (24 layers)                          │
│  ├── Self-Attention Heads: 16                           │
│  ├── Attention Head Size: 64                             │
│  ├── Intermediate Size: 2816                             │
│  └── Activation: SwiGLU                                  │
├─────────────────────────────────────────────────────────────┤
│  LoRA Adapter Layers                                     │
│  ├── Target Modules: q_proj, v_proj, k_proj, o_proj     │
│  ├── Target Modules: gate_proj, up_proj, down_proj      │
│  ├── LoRA Rank: 16                                       │
│  ├── LoRA Alpha: 32                                      │
│  └── LoRA Dropout: 0.05                                  │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                            │
│  ├── Linear Layer                                        │
│  └── Softmax for Next Token Prediction                   │
└─────────────────────────────────────────────────────────────┘
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                     # Rank: Number of new parameters
    lora_alpha=32,           # Alpha: Scaling factor (2x rank)
    target_modules=[         # Target modules for adaptation:
        "q_proj", "v_proj", "k_proj", "o_proj",    # Self-attention
        "gate_proj", "up_proj", "down_proj"        # Feed-forward
    ],
    lora_dropout=0.05,       # Dropout: Prevent overfitting
    bias="none",             # Don't modify bias
    task_type=TaskType.CAUSAL_LM  # Task type: Text generation
)
```

## 📊 Training Configuration

### Dataset
- **Source**: Indonesian Food Recipes Dataset
- **Format**: CSV with columns: Title, Ingredients, Steps
- **Data Augmentation**: 40+ training examples per recipe
- **Total Examples**: Variable based on dataset size

### Training Parameters
```python
training_args = TrainingArguments(
    num_train_epochs=3,                       # Training epochs
    per_device_train_batch_size=2,           # Batch size per device
    gradient_accumulation_steps=4,            # Gradient accumulation
    learning_rate=2e-5,                      # Learning rate
    weight_decay=0.01,                       # Weight decay
    warmup_steps=100,                        # Warmup steps
    fp16=True,                               # Mixed precision training
    gradient_checkpointing=True,             # Memory efficient
    label_smoothing_factor=0.1,              # Label smoothing
    save_steps=500,                          # Save checkpoint every 500 steps
    logging_steps=100,                       # Log every 100 steps
)
```

### Data Processing Pipeline
1. **Ingredient Cleaning**: Extract main ingredients from measurements
2. **Example Generation**: Create 40+ training examples per recipe
3. **Chat Template**: Format as conversation using Qwen chat template
4. **Tokenization**: Convert to tokens with max length 512
5. **Batching**: Process in batches with gradient accumulation

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers datasets pandas peft tensorboard
```

### Basic Testing
```bash
# Basic comprehensive test
python test_model_inference.py

# Interactive mode
python test_model_inference.py --interactive

# Full test suite with all features
python test_model_inference.py --benchmark --temperature-test --memory-test --save-results results.json

# Custom model path
python test_model_inference.py --model-path ./IFMF-Qwen2-0.5B-Instruct-full
```

### Quick Testing
```bash
# Quick test (default)
python quick_model_test.py

# Interactive mode
python quick_model_test.py --interactive
```

## 🏋️ Training

### Start Training
```bash
python training-v4-all-sample.py
```

### Training Features
- ✅ **Checkpoint Resume**: Automatically resume from latest checkpoint
- 📊 **TensorBoard Logging**: Monitor training progress
- 💾 **Auto Save**: Save checkpoints every 500 steps
- 🔄 **Gradient Accumulation**: Efficient memory usage
- ⚡ **Mixed Precision**: FP16 training for speed

### Training Data Augmentation
Each recipe generates 40+ training examples:
1. **Recipe Requests** (5 variations): Formal recipe requests
2. **Simple Questions** (15 variations): Casual recipe questions
3. **Ingredient Recommendations** (10 variations): Based on available ingredients
4. **Ingredient Inquiries** (10 variations): Questions about specific ingredients

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir ./IFMF-Qwen2.5-0.5B-Instruct-v4-small/tensorboard
```

### Training Metrics
- Loss curves
- Learning rate schedule
- Gradient norms
- Memory usage

## 🗂️ Model Output Structure

```
IFMF-Qwen2.5-0.5B-Instruct-v4-small/
├── config.json                    # Model configuration
├── pytorch_model.bin             # Model weights
├── tokenizer.json                # Tokenizer
├── tokenizer_config.json         # Tokenizer config
├── training_metadata.json        # Training metadata
├── checkpoint-500/               # Training checkpoints
├── checkpoint-1000/
├── logs/                         # Training logs
└── tensorboard/                  # TensorBoard logs
```

## 🤖 Model Capabilities

### 1. Recipe Generation
**Input**: "Bagaimana cara membuat rendang?"
**Output**: Complete recipe with ingredients, steps, and cooking tips

### 2. Ingredient-based Recommendations
**Input**: "Saya punya daging sapi, bisa masak apa?"
**Output**: Recipe recommendations based on available ingredients

### 3. Cooking Q&A
**Input**: "Tips memasak daging sapi gimana?"
**Output**: Detailed cooking tips and techniques

### 4. Simple Recipe Requests
**Input**: "Resep rendang dong!"
**Output**: Concise recipe with essential information

## 🔧 Technical Details

### Memory Optimization
- **Gradient Checkpointing**: Reduces memory usage
- **Mixed Precision**: FP16 training
- **LoRA**: Parameter efficient fine-tuning
- **Batch Processing**: Efficient data loading

### Performance Features
- **Auto Device Mapping**: Automatic GPU/CPU detection
- **Resume Training**: Continue from checkpoints
- **Progress Monitoring**: Real-time training metrics
- **Error Handling**: Robust error recovery

## 📚 Citation

If you use this model, please cite:

```bibtex
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
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Qwen Team for the base model
- Hugging Face for the transformers library
- Microsoft for LoRA technology
- Indonesian culinary community for recipe data

---

**🍳 Happy Cooking with AI Chef Indonesia! 🍳**