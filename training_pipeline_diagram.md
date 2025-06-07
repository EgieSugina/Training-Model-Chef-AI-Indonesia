# Training Pipeline Diagram for Qwen Indonesian Food Recipe Model

```mermaid
flowchart TD
    A[🚀 Initialize Training Process] --> B[🔧 Setup Environment & Imports]
    B --> C{📂 Check for Existing Checkpoints}
    
    C -->|Checkpoints Found| D[📥 Load Model from Latest Checkpoint]
    C -->|No Checkpoints| E[🤗 Load Base Qwen Model from HuggingFace]
    
    D --> F[📥 Load Tokenizer from Checkpoint]
    E --> G[🤗 Load Base Tokenizer from HuggingFace]
    
    F --> H[⚙️ Configure Model Settings]
    G --> H
    
    H --> I[🛠️ Set Pad Token ID]
    I --> J[🔧 Prepare Model for k-bit Training]
    J --> K[📋 Setup Base Model Config]
    K --> L[📁 Create Output Directory]
    
    L --> M[🎯 Configure LoRA Parameters]
    M --> N[🔗 Apply LoRA Adapters to Model]
    N --> O[⚡ Enable Input Gradients]
    
    O --> P[📊 Load Recipe Dataset CSV]
    P --> Q{📋 Validate Dataset Exists}
    Q -->|File Not Found| R[❌ Raise FileNotFoundError]
    Q -->|File Found| S[🔍 Analyze Dataset Structure]
    
    S --> T[🧹 Clean Ingredients Data]
    T --> U[🔄 Generate Multiple Training Examples]
    
    U --> V[📝 Create Standard Recipe Requests]
    U --> W[❓ Create Simple Recipe Questions]  
    U --> X[🥘 Create Ingredient-based Recommendations]
    U --> Y[🔍 Create Ingredient Inquiry Responses]
    
    V --> Z[💬 Apply Chat Templates]
    W --> Z
    X --> Z
    Y --> Z
    
    Z --> AA[🎯 Tokenize All Examples]
    AA --> BB[📦 Create Data Collator for LM]
    
    BB --> CC[⚙️ Setup Training Arguments]
    CC --> DD[📊 Configure TensorBoard Logging]
    DD --> EE[🏋️ Initialize Trainer Object]
    
    EE --> FF[🔄 Start Training Loop]
    FF --> GG{🎯 Training Successful?}
    
    GG -->|Success| HH[💾 Save Fine-tuned Model]
    GG -->|Failed| II[🚨 Handle Training Error]
    
    HH --> JJ[💾 Save Tokenizer]
    JJ --> KK[📊 Save Training Metadata]
    KK --> LL[📈 Close TensorBoard Writer]
    LL --> MM[✅ Training Complete Successfully]
    
    II --> NN[❌ Raise Exception & Stop]
    
    subgraph "🎯 LoRA Configuration Details"
        M1[📐 Rank: 16]
        M2[🔢 Alpha: 32]
        M3[🎯 Target Modules:<br/>q_proj, v_proj, k_proj,<br/>o_proj, gate_proj,<br/>up_proj, down_proj]
        M4[📉 Dropout: 0.05]
        M5[🚫 Bias: none]
        M --> M1
        M --> M2
        M --> M3
        M --> M4
        M --> M5
    end
    
    subgraph "📊 Dataset Processing Details"
        T1[🧹 Split Ingredients by '--']
        T2[🔤 Remove Numbers & Measurements]
        T3[📝 Extract Main Ingredient Names]
        T4[🔤 Convert to Lowercase]
        T --> T1
        T1 --> T2
        T2 --> T3
        T3 --> T4
    end
    
    subgraph "🔄 Training Example Generation"
        U1[📝 5 Standard Recipe Variations]
        U2[❓ 15 Simple Question Variations]
        U3[🥘 10 Ingredient Recommendation Variations]
        U4[🔍 10 Ingredient Inquiry Variations]
        U --> U1
        U --> U2
        U --> U3
        U --> U4
    end
    
    subgraph "⚙️ Training Configuration"
        CC1[🔄 Epochs: 3]
        CC2[📦 Batch Size: 2]
        CC3[📈 Learning Rate: 2e-5]
        CC4[🔁 Gradient Accumulation: 4]
        CC5[💾 Save Steps: 500]
        CC6[📊 Logging Steps: 100]
        CC7[⚖️ Weight Decay: 0.01]
        CC8[🔥 FP16: True (if CUDA)]
        CC9[✂️ Gradient Checkpointing: True]
        CC10[🎯 Label Smoothing: 0.1]
        CC11[🔥 Warmup Steps: 100]
        CC --> CC1
        CC --> CC2
        CC --> CC3
        CC --> CC4
        CC --> CC5
        CC --> CC6
        CC --> CC7
        CC --> CC8
        CC --> CC9
        CC --> CC10
        CC --> CC11
    end
    
    subgraph "💾 Output & Monitoring"
        KK1[📊 Model Name & Version]
        KK2[📈 Total Recipes Count]
        KK3[🔢 Total Training Examples]
        KK4[✨ Model Capabilities List]
        KK5[📊 TensorBoard Logs]
        KK6[💾 Checkpoint Management]
        KK --> KK1
        KK --> KK2
        KK --> KK3
        KK --> KK4
        KK --> KK5
        KK --> KK6
    end
    
    style A fill:#e3f2fd
    style MM fill:#c8e6c9
    style NN fill:#ffcdd2
    style R fill:#ffcdd2
    style FF fill:#fff3e0
    style HH fill:#f3e5f5
    style GG fill:#ffecb3
    style C fill:#e8f5e8
    style Q fill:#e8f5e8
```

## 🔍 Detailed Pipeline Components

### 1. 🚀 **Initialization & Setup**
- Import semua library yang diperlukan (torch, transformers, pandas, datasets, peft)
- Setup environment variables dan device detection (CUDA/CPU)
- Initialize TensorBoard writer untuk monitoring

### 2. 📂 **Model Loading Strategy**
- **Checkpoint Detection**: Scan output directory untuk checkpoint terbaru
- **Resume Logic**: Load dari checkpoint jika ada, otherwise load base model
- **Model Variants**: Support untuk Qwen 0.5B, 1.5B, atau 3B
- **Device Mapping**: Automatic device mapping untuk multi-GPU

### 3. 🎯 **LoRA Configuration (Parameter Efficient Fine-tuning)**
- **Rank 16**: Balance antara performance dan efisiensi
- **Alpha 32**: Scaling factor untuk LoRA weights
- **Target Modules**: All attention & MLP layers
- **Task Type**: Causal Language Modeling
- **Dropout**: 0.05 untuk regularization

### 4. 📊 **Advanced Data Processing**
- **CSV Validation**: Check file existence dan structure
- **Ingredient Cleaning**: Remove measurements, extract main ingredients
- **Multi-variation Generation**: 40+ variations per recipe
- **Chat Template Formatting**: Apply Qwen chat format
- **Tokenization**: Max length 512 tokens dengan padding

### 5. 🔄 **Training Example Types**
- **Standard Requests** (5 variations): Formal recipe requests
- **Simple Questions** (15 variations): Casual recipe inquiries  
- **Ingredient Recommendations** (10 variations): Suggest dishes from available ingredients
- **Ingredient Inquiries** (10 variations): What to cook with specific ingredients

### 6. ⚙️ **Comprehensive Training Configuration**
- **Memory Optimization**: FP16, gradient checkpointing, gradient accumulation
- **Learning Schedule**: Warmup steps, weight decay, label smoothing
- **Monitoring**: TensorBoard integration, detailed logging
- **Checkpointing**: Save every 500 steps, keep only 2 latest
- **Resume Capability**: Automatic resume from latest checkpoint

### 7. 💾 **Output Management**
- **Model Artifacts**: Fine-tuned model, tokenizer, configuration
- **Metadata**: Training statistics, model capabilities
- **Monitoring**: TensorBoard logs, training curves
- **Backup**: Checkpoint management dengan rotation

### 8. 🔄 **Error Handling & Recovery**
- **File Validation**: Check dataset availability  
- **Training Monitoring**: Success/failure detection
- **Exception Handling**: Graceful error management
- **Resume Logic**: Continue from interruption points

## 📈 **Model Capabilities After Training**
1. 🍽️ Memberikan resep lengkap masakan Indonesia
2. ❓ Menjawab pertanyaan sederhana tentang resep
3. 🥘 Merekomendasikan masakan berdasarkan bahan tersedia
4. 🔍 Memberikan saran penggunaan bahan tertentu
5. 💡 Tips dan trik memasak tradisional Indonesia 