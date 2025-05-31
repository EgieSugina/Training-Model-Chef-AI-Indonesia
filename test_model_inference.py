#!/usr/bin/env python3
"""
Model Inference Test Script for Home Chef AI
Tests the fine-tuned model from training-v4.py with various queries and scenarios.
"""

import torch
import time
import json
import os
import argparse
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ChefAIModelTester:
    """Test suite for the fine-tuned Chef AI model"""
    
    def __init__(self, model_path: str = "./IFMF-small", base_model: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to the fine-tuned model directory
            base_model: Base model name used for training
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        print(f"🔧 Initializing Chef AI Model Tester")
        print(f"📁 Model path: {model_path}")
        print(f"🖥️ Device: {self.device}")
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print("\n🔄 Loading model and tokenizer...")
            
            # Check if model directory exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # Load tokenizer
            print("📚 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model
            print("🧠 Loading base model...")
            base_model_loaded = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapters
            print("🔗 Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(base_model_loaded, self.model_path)
            
            # Move to device if needed
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            print(f"📊 Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def generate_response(self, user_message: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate response from the model
        
        Args:
            user_message: User input message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            # Prepare messages with system prompt
            messages = [
                {"role": "system", "content": "Kamu adalah Chef Indonesia yang ahli dalam masakan tradisional. Tugasmu adalah memberikan resep lengkap dengan bahan-bahan dan langkah memasak yang detail dan mudah diikuti."},
                {"role": "user", "content": user_message}
            ]
            
            # Format with chat template
            formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response (only new tokens)
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def test_predefined_queries(self):
        """Test model with predefined queries"""
        print("\n🧪 Testing with predefined queries...")
        
        test_queries = [
            # Recipe requests
            {
                "category": "Recipe Request",
                "query": "Bagaimana cara membuat nasi goreng spesial?",
                "expected_elements": ["bahan", "langkah", "nasi", "telur"]
            },
            {
                "category": "Recipe Request", 
                "query": "Tolong ajarkan saya resep rendang daging yang autentik.",
                "expected_elements": ["daging", "santan", "cabai", "bumbu"]
            },
            {
                "category": "Simple Question",
                "query": "Resep soto ayam dong!",
                "expected_elements": ["ayam", "kuah", "bumbu", "rebus"]
            },
            {
                "category": "Ingredient-based",
                "query": "Saya punya ayam dan kentang, bisa buat masakan apa?",
                "expected_elements": ["ayam", "kentang", "masakan", "resep"]
            },
            {
                "category": "Ingredient Inquiry",
                "query": "Saya ada santan kelapa, bisa untuk masak apa?",
                "expected_elements": ["santan", "masakan", "resep"]
            },
            {
                "category": "Traditional Recipe",
                "query": "Cara membuat gudeg Yogyakarta yang enak?",
                "expected_elements": ["nangka", "gudeg", "yogyakarta", "santan"]
            }
        ]
        
        results = []
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {test['category']} ---")
            print(f"Query: {test['query']}")
            
            start_time = time.time()
            response = self.generate_response(test['query'])
            end_time = time.time()
            
            print(f"Response: {response}")
            print(f"⏱️ Generation time: {end_time - start_time:.2f}s")
            
            # Check for expected elements
            found_elements = []
            for element in test['expected_elements']:
                if element.lower() in response.lower():
                    found_elements.append(element)
            
            score = len(found_elements) / len(test['expected_elements']) * 100
            print(f"📊 Relevance score: {score:.1f}% ({len(found_elements)}/{len(test['expected_elements'])} elements found)")
            
            results.append({
                "query": test['query'],
                "category": test['category'],
                "response": response,
                "generation_time": end_time - start_time,
                "relevance_score": score,
                "found_elements": found_elements
            })
        
        return results
    
    def benchmark_performance(self, num_tests: int = 5):
        """Benchmark model performance"""
        print(f"\n⚡ Running performance benchmark ({num_tests} tests)...")
        
        test_query = "Bagaimana cara membuat nasi goreng yang enak?"
        times = []
        
        # Warm up
        print("🔥 Warming up...")
        self.generate_response(test_query, max_tokens=100)
        
        print("🏃‍♂️ Running benchmark...")
        for i in range(num_tests):
            start_time = time.time()
            response = self.generate_response(test_query, max_tokens=256)
            end_time = time.time()
            
            generation_time = end_time - start_time
            times.append(generation_time)
            print(f"Test {i+1}: {generation_time:.2f}s")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Performance Results:")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Min time: {min_time:.2f}s")
        print(f"Max time: {max_time:.2f}s")
        print(f"Throughput: {256/avg_time:.1f} tokens/second")
        
        return {
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": 256/avg_time
        }
    
    def test_different_temperatures(self):
        """Test model behavior with different temperature settings"""
        print("\n🌡️ Testing different temperature settings...")
        
        query = "Resep nasi goreng sederhana untuk pemula?"
        temperatures = [0.1, 0.5, 0.7, 1.0]
        
        for temp in temperatures:
            print(f"\n--- Temperature: {temp} ---")
            response = self.generate_response(query, temperature=temp, max_tokens=200)
            print(f"Response: {response[:200]}...")
    
    def test_memory_usage(self):
        """Test memory usage during inference"""
        print("\n💾 Testing memory usage...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
            
            # Generate a response
            query = "Cara membuat rendang daging yang lezat?"
            response = self.generate_response(query)
            
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()
            
            print(f"Peak GPU memory: {peak_memory / 1024**2:.1f} MB")
            print(f"Final GPU memory: {final_memory / 1024**2:.1f} MB")
            print(f"Memory increase: {(final_memory - initial_memory) / 1024**2:.1f} MB")
            
            return {
                "initial_memory_mb": initial_memory / 1024**2,
                "peak_memory_mb": peak_memory / 1024**2,
                "final_memory_mb": final_memory / 1024**2
            }
        else:
            print("CPU mode - detailed memory tracking not available")
            return None
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🗣️ Interactive testing mode")
        print("Type your questions about Indonesian recipes. Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 User: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Chef AI: ", end="", flush=True)
                start_time = time.time()
                response = self.generate_response(user_input)
                end_time = time.time()
                
                print(response)
                print(f"⏱️ ({end_time - start_time:.2f}s)")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def save_test_results(self, results: List[Dict[str, Any]], filename: str = "test_results.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Test results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
    
    def generate_test_report(self, results: List[Dict[str, Any]]):
        """Generate a comprehensive test report"""
        print("\n📋 TEST REPORT")
        print("=" * 50)
        
        # Overall statistics
        total_tests = len(results)
        avg_score = sum(r['relevance_score'] for r in results) / total_tests
        avg_time = sum(r['generation_time'] for r in results) / total_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Average relevance score: {avg_score:.1f}%")
        print(f"Average generation time: {avg_time:.2f}s")
        
        # Category breakdown
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['relevance_score'])
        
        print(f"\n📊 Score by category:")
        for category, scores in categories.items():
            avg_cat_score = sum(scores) / len(scores)
            print(f"  {category}: {avg_cat_score:.1f}%")
        
        # Performance rating
        if avg_score >= 80:
            rating = "🌟 Excellent"
        elif avg_score >= 70:
            rating = "👍 Good"
        elif avg_score >= 60:
            rating = "⚠️ Fair"
        else:
            rating = "❌ Needs Improvement"
        
        print(f"\n🏆 Overall Rating: {rating}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the fine-tuned Chef AI model")
    parser.add_argument("--model-path", default="./IFMF-small", help="Path to the fine-tuned model")
    parser.add_argument("--base-model", default="Qwen/Qwen2-0.5B-Instruct", help="Base model name")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--temperature-test", action="store_true", help="Test different temperatures")
    parser.add_argument("--memory-test", action="store_true", help="Test memory usage")
    parser.add_argument("--save-results", help="Save results to file")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ChefAIModelTester(args.model_path, args.base_model)
    
    try:
        # Load model
        tester.load_model()
        
        # Run tests based on arguments
        if args.interactive:
            tester.interactive_test()
        else:
            # Run comprehensive test suite
            print("\n🚀 Starting comprehensive test suite...")
            
            # 1. Predefined queries test
            results = tester.test_predefined_queries()
            
            # 2. Generate report
            tester.generate_test_report(results)
            
            # 3. Optional tests
            if args.benchmark:
                benchmark_results = tester.benchmark_performance()
            
            if args.temperature_test:
                tester.test_different_temperatures()
            
            if args.memory_test:
                memory_results = tester.test_memory_usage()
            
            # 4. Save results if requested
            if args.save_results:
                tester.save_test_results(results, args.save_results)
            
            print("\n✅ All tests completed!")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 