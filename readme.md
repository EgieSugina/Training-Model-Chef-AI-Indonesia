# Basic comprehensive test
python test_model_inference.py

# Interactive mode
python test_model_inference.py --interactive

# Full test suite dengan semua fitur
python test_model_inference.py --benchmark --temperature-test --memory-test --save-results results.json

# Custom model path
python test_model_inference.py --model-path ./IFMF-Qwen2-0.5B-Instruct-full


# Quick test (default)
python quick_model_test.py

# Interactive mode
python quick_model_test.py --interactive