from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import os
import json
import warnings
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
OUTPUT_DIR = "IFMF-Qwen2-0.5B-Instruct-full"
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
USE_CUDA = torch.cuda.is_available()  # Auto-detect CUDA availability
DEVICE = "cuda" if USE_CUDA else "cpu"

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store active generation tasks
active_generations = {}

# Load model based on device configuration
try:
    print(f"Loading model on {DEVICE}...")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Configure model loading based on device
    if DEVICE == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    # Ensure model is in evaluation mode
    # model.eval()
    
    print(f"Model loaded successfully on {DEVICE}")
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
    - top_p: Top-p sampling (default 0.9)
    """
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = max(10, min(data.get("max_tokens", 512), 1024))  # Clamp tokens
        temperature = max(0.1, min(data.get("temperature", 0.7), 2.0))  # Clamp temperature
        top_p = max(0.1, min(data.get("top_p", 0.9), 1.0))  # Clamp top_p
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Generate a unique ID for this request
        request_id = id(request)
        # Create a cancellation event
        cancel_event = asyncio.Event()
        # Store in active generations
        active_generations[request_id] = cancel_event
        
        # Format messages with system prompt for Indonesian chef
        messages = [
            {"role": "system", "content": "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional. Tugasmu adalah memberikan resep lengkap dengan bahan-bahan dan langkah memasak yang detail dan mudah diikuti."},
            {"role": "user", "content": prompt}
        ]
        
        # Convert messages to model input format
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        async def generate_stream():
            try:
                # Prepare input with attention mask
                inputs = tokenizer(
                    formatted_prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                
                # Move inputs to the same device as model
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with chunks to allow streaming
                generate_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                }
                
                # For streaming in chunks
                chunk_size = 8  # Generate chunks of this size
                total_new_tokens = 0
                generated_so_far = ""
                
                # Check if request has been cancelled
                if cancel_event.is_set():
                    yield f"data: {json.dumps({'text': '[Generation cancelled]'})}\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                
                # Initial generation
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        streamer=None
                    )
                
                # Get the full generated text
                full_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                # Stream the output in chunks
                for i in range(0, len(full_output), 10):
                    # Check if request has been cancelled
                    if cancel_event.is_set():
                        yield f"data: {json.dumps({'text': '[Generation cancelled]'})}\n\n"
                        break
                    
                    chunk = full_output[i:i+10]
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    
                    # Small delay to allow cancellation to be processed
                    await asyncio.sleep(0.01)
                    
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                print(f"Generation error: {e}")
            finally:
                # Clean up
                if request_id in active_generations:
                    del active_generations[request_id]
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel/{request_id}")
async def cancel_generation(request_id: str):
    """Cancel an ongoing generation request"""
    try:
        request_id = int(request_id)
        if request_id in active_generations:
            # Set the cancellation event
            active_generations[request_id].set()
            return {"status": "cancelled", "request_id": request_id}
        else:
            raise HTTPException(status_code=404, detail="Generation request not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request ID format")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "device": DEVICE}

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)