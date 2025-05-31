from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Disable BitsAndBytes
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BNB_CUDA_VERSION"] = "0"

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with CPU-compatible configuration
try:
    model_path = "IFMF-Qwen2.5-1.5B-Instruct"
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Load model with most simple approach (confirmed working)
    print("Loading model on CPU (this will take some time)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True,
    )
    
    print(f"Model loaded successfully: {model.__class__.__name__}")
    print(f"Model parameters: {model.num_parameters():,}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.post("/generate")
async def generate(request: Request):
    """
    Generate text with streaming response.
    
    Expects JSON payload with:
    - prompt: Input text
    - max_tokens: Maximum tokens to generate (default 512)
    - temperature: Sampling temperature (default 0.7)
    """
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = max(10, min(data.get("max_tokens", 1024), 1024))  # Clamp tokens
        temperature = max(0.1, min(data.get("temperature", 0.7), 2.0))  # Clamp temperature
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Format messages with system prompt for Indonesian chef
        messages = [
            {"role": "system", "content": "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional. Tugasmu adalah memberikan resep lengkap dengan bahan-bahan dan langkah memasak yang detail dan mudah diikuti."},
            {"role": "user", "content": prompt}
        ]
        
        # Convert messages to model input format
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        def generate_stream():
            try:
                # Prepare input with attention mask
                inputs = tokenizer(
                    formatted_prompt, 
                    return_tensors="pt",
                    padding=True,  # Add padding
                    truncation=True,  # Truncate if too long
                )
                
                # Add attention mask if missing
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                
                # Generate with chunks to allow streaming
                generate_kwargs = {
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "pad_token_id": tokenizer.eos_token_id,  # Use EOS as PAD
                }
                
                # For streaming in chunks
                chunk_size = 8  # Generate chunks of this size
                total_new_tokens = 0
                
                # Setup for generation
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                generated_so_far = ""
                
                while total_new_tokens < max_tokens:
                    # Generate next chunk
                    current_max_tokens = min(chunk_size, max_tokens - total_new_tokens)
                    
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + current_max_tokens,
                        **generate_kwargs
                    )
                    
                    # Extract only the newly generated tokens
                    new_tokens = outputs[0][input_ids.shape[1]:]
                    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Update for next iteration
                    input_ids = outputs
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones((attention_mask.shape[0], new_tokens.shape[0]), dtype=attention_mask.dtype)
                    ], dim=1)
                    
                    # Track progress
                    total_new_tokens += new_tokens.shape[0]
                    
                    # Yield the new text
                    yield f"data: {json.dumps({'text': decoded})}\n\n"
                    
                    # If generation ended early
                    if new_tokens.shape[0] < current_max_tokens:
                        break
            
            except Exception as e:
                print(f"Generation error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        print(f"Request processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)