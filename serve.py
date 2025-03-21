import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import threading
import time

# Define model and tokenizer
model_id = "kalisai/Nusantara-1.8B-Indo-Chat"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "mistralai/Mistral-Nemo-Instruct-2407"

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")

print(f"Starting application on device: {device}")
print(f"Using model: {model_id}")
print(f"HF token available: {'Yes' if hf_token else 'No'}")

# Load model and tokenizer with token if available
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    token=hf_token if hf_token else None
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    token=hf_token if hf_token else None
)
print("Model and tokenizer loaded successfully!")

# Set pad_token_id explicitly if it's not set
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
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id
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
    title="Asisten Memasak Indonesia",
    description="Asisten virtual untuk membantu Anda dengan masakan Indonesia",
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
    
    # Launch with the specified port
    demo.launch(server_name="0.0.0.0", server_port=port)

# =====================================================
# DEPLOYMENT STEPS FOR GOOGLE CLOUD RUN (SOUTHEAST ASIA)
# =====================================================
# 
# 1. Make sure you have the following files in your project directory:
#    - food.py (this file)
#    - requirements.txt
#    - Dockerfile
#    - .dockerignore
#
# 2. Install and set up Google Cloud SDK (if not already done):
#    https://cloud.google.com/sdk/docs/install
#
# 3. Authenticate with Google Cloud:
#    $ gcloud auth login
#
# 4. Set your Google Cloud project:
#    $ gcloud config set project [YOUR-PROJECT-ID]
#
# 5. Build your Docker image:
#    $ docker build -t gcr.io/[YOUR-PROJECT-ID]/indonesian-cooking-assistant .
#
# 6. Push the image to Google Container Registry:
#    $ gcloud auth configure-docker
#    $ docker push gcr.io/[YOUR-PROJECT-ID]/indonesian-cooking-assistant
#
# 7. Deploy to Cloud Run in Southeast Asia (Singapore):
#    $ gcloud run deploy indonesian-cooking-assistant \
#      --image gcr.io/[YOUR-PROJECT-ID]/indonesian-cooking-assistant \
#      --platform managed \
#      --region asia-southeast1 \
#      --allow-unauthenticated \
#      --memory 2Gi
#
# 8. Access your deployed application at the URL provided after deployment
#
# Note: You may need to adjust memory allocation based on model requirements.
# For the Nusantara-1.8B model, 2GB should be sufficient, but you can increase if needed.
