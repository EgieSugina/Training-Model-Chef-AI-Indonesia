# # Use a pipeline as a high-level helper
# from transformers import pipeline

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="kalisai/Nusantara-1.8b-Indo-Chat")
# pipe(messages)

# Use a pipeline as a high-level helper
from transformers import pipeline
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct-1M")
pipe(messages)