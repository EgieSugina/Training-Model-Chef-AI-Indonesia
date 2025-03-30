from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gradio as gr
import os
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Disable BitsAndBytes welcome message
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# Define model path
MODEL_PATH = "./IFMF-Qwen-0.5B-500food"

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load model with GPU configuration if available
try:
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Configure quantization if using CUDA
    quantization_config = None
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        logger.info("Using 8-bit quantization for CUDA")

    # Load model with appropriate device configuration
    logger.info(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    logger.info(f"Model loaded successfully: {model.__class__.__name__}")
    logger.info(f"Model parameters: {model.num_parameters():,}")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

def generate_text(prompt, max_tokens=256, temperature=0.7):
    """
    Generate text based on the input prompt with GPU acceleration when available.
    """
    try:
        # Prepare input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Add attention mask if missing
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # Generate text with optimized inference
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "num_return_sequences": 1,
            "pad_token_id": tokenizer.eos_token_id,
        }

        with torch.inference_mode():  # Disables gradient computation for speedup
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return only the newly generated part (remove the prompt)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return generated_text[len(tokenizer.decode(prompt_tokens, skip_special_tokens=True)):]

    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        return f"Error generating text: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Slider(minimum=10, maximum=1024, value=256, step=1, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(lines=10, label="Generated Text"),
    title="Food Recipe Generator",
    description="Generate food recipes using the IFMF-Qwen-0.5B-500food model",
    examples=[
        ["Write a recipe for chicken curry:", 500, 0.7],
        ["How do I make chocolate chip cookies?", 500, 0.7],
        ["Create a vegetarian pasta dish:", 500, 0.7]
    ]
)

# Add health check endpoint
@demo.app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_PATH, "device": device}

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Gradio server on port {port}...")
    
    # Launch with the specified port
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
