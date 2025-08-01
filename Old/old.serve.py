import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Starting application on device: {device}")

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "kalisai/Nusantara-1.8B-Indo-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("kalisai/Nusantara-1.8B-Indo-Chat")
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B") 
print("Model and tokenizer loaded successfully!")

# Set pad_token_id to eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# System message for the assistant
system_message = "Kamu adalah Chef Indonesia, asisten AI yang membantu membuat resep masakan Indonesia. Selalu berikan instruksi step by step yang jelas dan terperinci untuk setiap resep yang kamu bagikan. Pastikan langkah-langkah diurutkan dengan baik dan mudah diikuti oleh pengguna."

def predict(message, history):
    # Format conversation history
    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate response
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # Ensure attention_mask is passed to generate
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512
    )
    
    # Extract only the new tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    predict,
    title="Nusantara AI Assistant",
    description="Asisten virtual berbahasa Indonesia",
    theme="soft"
)

# Add a health check endpoint
@demo.app.get("/health")
def health_check():
    return {"status": "healthy"}

# Launch the app
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Gradio server on port {port}...")
    demo.launch(server_port=port)
