import os
import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "./indonesian-food-model-final-Qwen-7B-500f"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RecipeAssistant:
    def __init__(self, model_path: str):
        """
        Initialize the Recipe Assistant with model and tokenizer
        """
        self.model_path = model_path
        self.device = DEVICE
        
        # System message for consistent context
        self.system_message = (
            "Kamu adalah Chef Indonesia, asisten AI yang membantu membuat resep masakan Indonesia. "
            "Selalu berikan instruksi step by step yang jelas dan terperinci untuk setiap resep. "
            "Pastikan langkah-langkah diurutkan dengan baik dan mudah diikuti."
        )
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """
        Load model with optimized configuration
        """
        try:
            # Quantization configuration for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ) if self.device == "cuda" else None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(
        self, 
        message: str, 
        history: List[Tuple[str, str]], 
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response based on conversation history
        """
        try:
            # Prepare chat history
            messages = [{"role": "system", "content": self.system_message}]
            
            # Add previous conversation turns
            for user_msg, assistant_msg in history:
                messages.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ])
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            model_inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                add_special_tokens=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Extract and decode response
            response = self.tokenizer.decode(
                generated_ids[0][len(model_inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Maaf, terjadi kesalahan saat memproses permintaan Anda."

def create_gradio_interface(assistant: RecipeAssistant):
    """
    Create Gradio interface for the Recipe Assistant
    """
    demo = gr.ChatInterface(
        fn=assistant.generate_response,
        title="Asisten Resep Masakan Indonesia",
        description=(
            "Asisten AI untuk membantu Anda membuat resep masakan Indonesia. "
            "Tanyakan apapun tentang resep, teknik memasak, atau tips kuliner!"
        ),
        theme="soft",
        additional_inputs=[
            gr.Slider(
                minimum=100, 
                maximum=1024, 
                value=512, 
                label="Panjang Maksimal Respon"
            )
        ]
    )
    
    # Add health check endpoint
    @demo.app.get("/health")
    def health_check():
        return {"status": "healthy", "model": MODEL_PATH}
    
    return demo

def main():
    """
    Main application entry point
    """
    try:
        # Initialize Recipe Assistant
        recipe_assistant = RecipeAssistant(MODEL_PATH)
        
        # Create Gradio interface
        demo = create_gradio_interface(recipe_assistant)
        
        # Get port from environment or use default
        port = int(os.environ.get("PORT", 8080))
        
        # Launch Gradio app
        logger.info(f"Starting Gradio server on port {port}...")
        demo.launch(
            server_name="0.0.0.0", 
            server_port=port,
            show_error=True,
            share=True
        )
    
    except Exception as e:
        logger.error(f"Application startup failed: {e}")

if __name__ == "__main__":
    main()

# Requirements:
# pip install torch transformers gradio accelerate bitsandbytes
# pip install -U "transformers==4.37.2"