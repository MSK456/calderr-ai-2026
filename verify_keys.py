from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

print("GROQ Key:", groq_key[:8] if groq_key else "NOT FOUND")
print("HF Token:", hf_token[:8] if hf_token else "NOT FOUND")