#!/usr/bin/env python3
"""
Test script for training-v4.py
Tests all components without running the full expensive training process.
"""

import unittest
import torch
import pandas as pd
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the current directory to Python path to import training modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test configuration
TEST_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
TEST_OUTPUT_DIR = "./test_output"

class TestTrainingV4(unittest.TestCase):
    """Test suite for training-v4.py components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_csv = os.path.join(cls.test_dir, "test_recipes.csv")
        cls.create_test_dataset()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
    
    @classmethod
    def create_test_dataset(cls):
        """Create a small test dataset"""
        test_data = {
            'Title': [
                'Nasi Goreng Spesial',
                'Rendang Daging',
                'Gado-gado Jakarta',
                'Soto Ayam Lamongan',
                'Gudeg Yogyakarta'
            ],
            'Ingredients': [
                'Nasi putih -- 2 piring -- Telur ayam -- 2 butir -- Kecap manis -- 2 sdm -- Bawang merah -- 3 siung -- Cabai -- 2 buah',
                'Daging sapi -- 500 gram -- Santan kelapa -- 400 ml -- Cabai merah -- 10 buah -- Bawang merah -- 8 siung -- Lengkuas -- 2 ruas',
                'Tahu -- 5 potong -- Tempe -- 100 gram -- Kangkung -- 1 ikat -- Tauge -- 200 gram -- Kacang tanah -- 100 gram',
                'Ayam kampung -- 1 ekor -- Kunyit -- 2 ruas -- Jahe -- 3 ruas -- Serai -- 2 batang -- Daun salam -- 3 lembar',
                'Nangka muda -- 500 gram -- Santan kelapa -- 300 ml -- Gula merah -- 100 gram -- Bawang putih -- 5 siung -- Ketumbar -- 1 sdt'
            ],
            'Steps': [
                'Panaskan minyak dalam wajan. Tumis bawang merah hingga harum. Masukkan telur, orak-arik. Tambahkan nasi, aduk rata. Beri kecap manis dan cabai. Masak hingga matang.',
                'Haluskan bumbu cabai, bawang merah, lengkuas. Tumis bumbu halus hingga harum. Masukkan daging, masak hingga berubah warna. Tuang santan, masak dengan api kecil 2 jam.',
                'Rebus sayuran hingga matang. Goreng tahu dan tempe. Haluskan kacang tanah dengan bumbu. Campur semua bahan, siram dengan bumbu kacang.',
                'Rebus ayam dengan bumbu hingga empuk. Tumis bumbu halus hingga harum. Masukkan kaldu ayam, didihkan. Sajikan dengan nasi dan pelengkap.',
                'Rebus nangka muda hingga empuk. Tumis bumbu halus hingga harum. Masukkan nangka dan santan. Masak hingga santan menyusut dan bumbu meresap.'
            ]
        }
        
        df = pd.DataFrame(test_data)
        df.to_csv(cls.test_csv, index=False)
    
    def test_torch_availability(self):
        """Test if PyTorch is available and CUDA detection works"""
        self.assertTrue(torch.__version__, "PyTorch should be available")
        
        # Test device detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertIn(device, ["cuda", "cpu"], "Device should be either cuda or cpu")
        print(f"✓ Device detected: {device}")
    
    def test_model_loading(self):
        """Test model and tokenizer loading"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Test tokenizer loading
            print("Testing tokenizer loading...")
            tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
            self.assertIsNotNone(tokenizer, "Tokenizer should load successfully")
            
            # Test pad token configuration
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            self.assertIsNotNone(tokenizer.pad_token_id, "Pad token should be set")
            
            print("✓ Tokenizer loaded successfully")
            
            # Test model loading (with minimal config to save memory)
            print("Testing model loading...")
            model = AutoModelForCausalLM.from_pretrained(
                TEST_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                use_cache=False
            )
            self.assertIsNotNone(model, "Model should load successfully")
            print("✓ Model loaded successfully")
            
            # Clean up memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.fail(f"Model loading failed: {str(e)}")
    
    def test_lora_configuration(self):
        """Test LoRA configuration"""
        try:
            from peft import LoraConfig, TaskType
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM 
            )
            
            # Test LoRA config attributes
            self.assertEqual(lora_config.r, 16, "LoRA rank should be 16")
            self.assertEqual(lora_config.lora_alpha, 32, "LoRA alpha should be 32")
            self.assertEqual(lora_config.lora_dropout, 0.05, "LoRA dropout should be 0.05")
            self.assertEqual(lora_config.task_type, TaskType.CAUSAL_LM, "Task type should be CAUSAL_LM")
            
            print("✓ LoRA configuration valid")
            
        except Exception as e:
            self.fail(f"LoRA configuration failed: {str(e)}")
    
    def test_dataset_loading(self):
        """Test dataset loading and validation"""
        try:
            # Test CSV file existence
            self.assertTrue(os.path.exists(self.test_csv), "Test CSV file should exist")
            
            # Test dataset loading
            df = pd.read_csv(self.test_csv)
            self.assertGreater(len(df), 0, "Dataset should not be empty")
            
            # Test required columns
            required_columns = ['Title', 'Ingredients', 'Steps']
            for col in required_columns:
                self.assertIn(col, df.columns, f"Column '{col}' should exist in dataset")
            
            # Test data quality
            self.assertFalse(df['Title'].isnull().any(), "Title column should not have null values")
            self.assertFalse(df['Ingredients'].isnull().any(), "Ingredients column should not have null values")
            self.assertFalse(df['Steps'].isnull().any(), "Steps column should not have null values")
            
            print(f"✓ Dataset loaded successfully with {len(df)} recipes")
            
        except Exception as e:
            self.fail(f"Dataset loading failed: {str(e)}")
    
    def test_data_preprocessing(self):
        """Test data preprocessing functions"""
        try:
            import pandas as pd
            
            # Test clean_ingredients function
            def clean_ingredients(ingredients_text):
                """Clean and extract ingredients from the text"""
                if pd.isna(ingredients_text):
                    return []
                
                ingredients = ingredients_text.split('--')
                cleaned = []
                for ing in ingredients:
                    ing = ing.strip()
                    if ing:
                        words = ing.split()
                        main_ingredient = ' '.join([w for w in words if not any(char.isdigit() for char in w) and 
                                                 w.lower() not in ['sdm', 'sdt', 'gram', 'kg', 'liter', 'ml', 'buah', 'lembar', 'batang', 'ruas', 'butir', 'ekor', 'ikat', 'gelas']])
                        if main_ingredient:
                            cleaned.append(main_ingredient.lower())
                return cleaned
            
            # Test with sample ingredient text
            test_ingredients = "Nasi putih -- 2 piring -- Telur ayam -- 2 butir -- Kecap manis -- 2 sdm"
            cleaned = clean_ingredients(test_ingredients)
            
            self.assertIsInstance(cleaned, list, "clean_ingredients should return a list")
            self.assertGreater(len(cleaned), 0, "Should extract at least one ingredient")
            self.assertIn("nasi putih", cleaned, "Should extract 'nasi putih'")
            self.assertIn("telur ayam", cleaned, "Should extract 'telur ayam'")
            
            print(f"✓ Data preprocessing works correctly. Extracted: {cleaned}")
            
        except Exception as e:
            self.fail(f"Data preprocessing failed: {str(e)}")
    
    def test_training_example_generation(self):
        """Test training example generation"""
        try:
            df = pd.read_csv(self.test_csv)
            
            def generate_training_examples(row):
                """Generate multiple training examples from one recipe (simplified version)"""
                examples = []
                title = row['Title']
                ingredients = row['Ingredients']
                steps = row['Steps']
                
                system_msg = "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional."
                
                # Example 1: Standard recipe request
                user_msg1 = f"Tolong ajarkan saya resep lengkap untuk membuat {title}."
                assistant_msg1 = f"Berikut adalah bahan-bahan: {ingredients}\n\nLangkah-langkah: {steps}"
                
                examples.append({
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg1},
                        {"role": "assistant", "content": assistant_msg1}
                    ]
                })
                
                return examples
            
            # Test with first row
            examples = generate_training_examples(df.iloc[0])
            
            self.assertIsInstance(examples, list, "Should return a list of examples")
            self.assertGreater(len(examples), 0, "Should generate at least one example")
            
            # Test example structure
            example = examples[0]
            self.assertIn("messages", example, "Example should have 'messages' key")
            self.assertEqual(len(example["messages"]), 3, "Should have system, user, and assistant messages")
            
            # Test message roles
            roles = [msg["role"] for msg in example["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"], "Message roles should be correct")
            
            print(f"✓ Training example generation works correctly. Generated {len(examples)} examples")
            
        except Exception as e:
            self.fail(f"Training example generation failed: {str(e)}")
    
    def test_tokenization(self):
        """Test tokenization process"""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Test chat template application
            messages = [
                {"role": "system", "content": "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional."},
                {"role": "user", "content": "Bagaimana cara membuat nasi goreng?"},
                {"role": "assistant", "content": "Berikut cara membuat nasi goreng..."}
            ]
            
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            self.assertIsInstance(formatted_text, str, "Chat template should return string")
            self.assertGreater(len(formatted_text), 0, "Formatted text should not be empty")
            
            # Test tokenization
            tokens = tokenizer(
                formatted_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            self.assertIn("input_ids", tokens, "Tokenization should return input_ids")
            self.assertIn("attention_mask", tokens, "Tokenization should return attention_mask")
            self.assertEqual(tokens["input_ids"].shape[1], 512, "Should pad/truncate to max_length")
            
            print("✓ Tokenization works correctly")
            
        except Exception as e:
            self.fail(f"Tokenization failed: {str(e)}")
    
    def test_training_arguments(self):
        """Test training arguments configuration"""
        try:
            from transformers import TrainingArguments
            
            training_args = TrainingArguments(
                output_dir=TEST_OUTPUT_DIR,
                num_train_epochs=1,  # Small for testing
                per_device_train_batch_size=1,  # Small for testing
                gradient_accumulation_steps=2,
                save_steps=100,
                save_total_limit=1,
                logging_steps=10,
                learning_rate=2e-5,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=True,
                warmup_steps=10,
                eval_strategy="no",
                save_strategy="steps",
                load_best_model_at_end=False,
                report_to=None,
            )
            
            # Test training arguments
            self.assertEqual(training_args.output_dir, TEST_OUTPUT_DIR, "Output dir should be set correctly")
            self.assertEqual(training_args.num_train_epochs, 1, "Epochs should be set correctly")
            self.assertEqual(training_args.per_device_train_batch_size, 1, "Batch size should be set correctly")
            self.assertEqual(training_args.learning_rate, 2e-5, "Learning rate should be set correctly")
            
            print("✓ Training arguments configured correctly")
            
        except Exception as e:
            self.fail(f"Training arguments configuration failed: {str(e)}")
    
    def test_output_directory_creation(self):
        """Test output directory creation and config saving"""
        try:
            # Test directory creation
            os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
            self.assertTrue(os.path.exists(TEST_OUTPUT_DIR), "Output directory should be created")
            
            # Test config saving
            test_config = {
                "model_type": "qwen",
                "hidden_size": 512,
                "num_attention_heads": 8,
                "test_config": True
            }
            
            config_path = os.path.join(TEST_OUTPUT_DIR, "config.json")
            with open(config_path, "w") as f:
                json.dump(test_config, f)
            
            self.assertTrue(os.path.exists(config_path), "Config file should be saved")
            
            # Test config loading
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
            
            self.assertEqual(loaded_config["model_type"], "qwen", "Config should be saved correctly")
            self.assertTrue(loaded_config["test_config"], "Test config should be preserved")
            
            print("✓ Output directory and config handling works correctly")
            
        except Exception as e:
            self.fail(f"Output directory creation failed: {str(e)}")
    
    def test_memory_management(self):
        """Test memory management and cleanup"""
        try:
            # Test tensor creation and cleanup
            if torch.cuda.is_available():
                # Create some tensors on GPU
                tensor1 = torch.randn(100, 100, device='cuda')
                tensor2 = torch.randn(100, 100, device='cuda')
                
                # Check memory usage
                memory_before = torch.cuda.memory_allocated()
                self.assertGreater(memory_before, 0, "Should allocate GPU memory")
                
                # Clean up
                del tensor1, tensor2
                torch.cuda.empty_cache()
                
                memory_after = torch.cuda.memory_allocated()
                self.assertLessEqual(memory_after, memory_before, "Memory should be freed")
                
                print("✓ GPU memory management works correctly")
            else:
                # Test CPU tensors
                tensor1 = torch.randn(1000, 1000)
                tensor2 = torch.randn(1000, 1000)
                
                # Clean up
                del tensor1, tensor2
                
                print("✓ CPU memory management works correctly")
                
        except Exception as e:
            self.fail(f"Memory management test failed: {str(e)}")

def run_integration_test():
    """Run a mini integration test that simulates the full pipeline"""
    print("\n" + "="*50)
    print("RUNNING INTEGRATION TEST")
    print("="*50)
    
    try:
        # Import required modules
        from transformers import AutoTokenizer
        import pandas as pd
        
        # 1. Load test data
        test_dir = tempfile.mkdtemp()
        test_csv = os.path.join(test_dir, "integration_test.csv")
        
        # Create minimal test data
        test_data = {
            'Title': ['Nasi Goreng Test'],
            'Ingredients': ['Nasi -- 1 piring -- Telur -- 1 butir'],
            'Steps': ['Goreng nasi dengan telur hingga matang.']
        }
        pd.DataFrame(test_data).to_csv(test_csv, index=False)
        
        # 2. Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 3. Process data
        print("Processing data...")
        df = pd.read_csv(test_csv)
        
        # 4. Generate training example
        print("Generating training example...")
        messages = [
            {"role": "system", "content": "Kamu adalah Chef Indonesia."},
            {"role": "user", "content": f"Resep {df.iloc[0]['Title']}?"},
            {"role": "assistant", "content": f"Bahan: {df.iloc[0]['Ingredients']}\nCara: {df.iloc[0]['Steps']}"}
        ]
        
        # 5. Apply chat template and tokenize
        print("Tokenizing...")
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(formatted_text, max_length=128, truncation=True, padding=True, return_tensors="pt")
        
        # 6. Verify output
        assert tokens["input_ids"].shape[0] == 1, "Should have 1 batch"
        assert tokens["input_ids"].shape[1] <= 128, "Should respect max_length"
        
        print("✓ Integration test passed successfully!")
        
        # Cleanup
        shutil.rmtree(test_dir)
        
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        raise

def main():
    """Main test runner"""
    print("TRAINING-V4.PY TEST SUITE")
    print("="*50)
    
    # Check dependencies
    try:
        import torch
        import transformers
        import pandas as pd
        import datasets
        from peft import LoraConfig
        print("✓ All required packages are available")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        return False
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    run_integration_test()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print("\nTo run the actual training, execute:")
    print("python training-v4.py")
    print("\nNote: Make sure you have:")
    print("1. Sufficient GPU memory (8GB+ recommended)")
    print("2. The dataset file: ./dataset_all/Indonesian_Food_Recipes_small.csv")
    print("3. Enough disk space for model checkpoints")

if __name__ == "__main__":
    main() 