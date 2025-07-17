import os
from huggingface_hub import InferenceClient
print(os.getenv("HF_TOKEN"))
print("ola")
from huggingface_hub import login
login("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")

client = InferenceClient(model="google/flan-t5-base", token="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")
try:
    response = client.text_generation("Test prompt")
    print(response)
except Exception as e:
    print(f"Hugging Face API error: {str(e)}")