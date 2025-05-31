#!/usr/bin/env python3
"""
Quick Model Test Script for Home Chef AI
Simple and fast testing for basic model validation.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")

def load_model(model_path="./IFMF-small", base_model="Qwen/Qwen2-0.5B-Instruct"):
    """Load the fine-tuned model"""
    print("🔄 Loading model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    base_model_loaded = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model_loaded, model_path)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer, device

def generate_response(model, tokenizer, device, user_message, max_tokens=256):
    """Generate response from the model"""
    messages = [
        {"role": "system", "content": "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional."},
        {"role": "user", "content": user_message}
    ]
    
    # Format and tokenize
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def quick_test():
    """Run quick test with predefined queries"""
    try:
        # Load model
        model, tokenizer, device = load_model()
        
        # Test queries
        test_queries = [
            "Bagaimana cara membuat nasi goreng?",
            "Resep rendang daging yang enak?",
            "Saya punya ayam, bisa masak apa?",
            "Cara membuat soto ayam?"
        ]
        
        print("\n🧪 Quick Test Results:")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            
            start_time = time.time()
            response = generate_response(model, tokenizer, device, query)
            end_time = time.time()
            
            print(f"   Response: {response}")
            print(f"   Time: {end_time - start_time:.2f}s")
        
        print("\n✅ Quick test completed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def interactive_mode():
    """Simple interactive mode"""
    try:
        model, tokenizer, device = load_model()
        
        print("\n🗣️ Interactive Mode - Ask me about Indonesian recipes!")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input:
                continue
            
            print("🤖 Chef AI: ", end="", flush=True)
            response = generate_response(model, tokenizer, device, user_input)
            print(response)
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        quick_test() 