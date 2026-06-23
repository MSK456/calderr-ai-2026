"""
Day 1: LLM Foundations
Goal: Explore different Groq models and understand temperature
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# EXPERIMENT 1: Compare different models
# ─────────────────────────────────────────────

def compare_models():
    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    question = "Explain what a transformer neural network is in exactly 2 sentences."

    print("\n" + "="*60)
    print("🔬 EXPERIMENT 1: Model Comparison")
    print("="*60)

    for model in models:
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=150
        )
        elapsed = time.time() - start

        print(f"\n📦 Model: {model}")
        print(f"⏱️  Speed: {elapsed:.2f}s")
        print(f"🪙  Tokens: {response.usage.total_tokens}")
        print(f"💬  Response: {response.choices[0].message.content}")
        print("-"*60)


# ─────────────────────────────────────────────
# EXPERIMENT 2: Temperature effect
# ─────────────────────────────────────────────

def explore_temperature():
    temperatures = [0, 0.5, 1.0, 1.5, 2.0]
    question = "Write one creative sentence about artificial intelligence."

    print("\n" + "="*60)
    print("🌡️  EXPERIMENT 2: Temperature Exploration")
    print("="*60)
    print("Observation: Same question, different temperatures\n")

    for temp in temperatures:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": question}],
            temperature=temp,
            max_tokens=80
        )
        print(f"Temp {temp}: {response.choices[0].message.content.strip()}")


# ─────────────────────────────────────────────
# EXPERIMENT 3: Context window test
# ─────────────────────────────────────────────

def test_context_window():
    print("\n" + "="*60)
    print("📐 EXPERIMENT 3: Context Window (Multi-turn Memory)")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Remember everything said."},
        {"role": "user", "content": "My name is MSK and I love robotics."},
        {"role": "assistant", "content": "Great to meet you MSK! Robotics is fascinating."},
        {"role": "user", "content": "What is my name and what do I love?"}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0,
        max_tokens=100
    )

    print(f"Model remembered: {response.choices[0].message.content}")
    print("\n✅ Key insight: All messages are sent together each time!")
    print("   This is how 'memory' works — it's just the full history resent.")


if __name__ == "__main__":
    compare_models()
    explore_temperature()
    test_context_window()
    print("\n✅ Day 1 Complete! You now understand models, temperature, and context.")