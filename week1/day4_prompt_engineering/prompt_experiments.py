"""
Day 4: Prompt Engineering — System Prompts, Few-Shot, CoT
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)
parser = StrOutputParser()


def test_prompt(label: str, system: str, user: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user)
    ])
    chain = prompt | llm | parser
    result = chain.invoke({})
    print(f"\n{'─'*50}")
    print(f"📌 {label}")
    print(f"{'─'*50}")
    print(result)


task = "Summarize this news: Scientists discovered a new species of deep-sea fish near the Mariana Trench. The fish can survive at depths of 8,000 meters and produces its own light."

print("="*60)
print("🧪 EXPERIMENT 1: System Prompt Effect")
print("="*60)

test_prompt(
    "No system prompt (baseline)",
    "You are a helpful assistant.",
    task
)

test_prompt(
    "Journalist persona",
    "You are a veteran journalist. Write summaries in AP style: lead sentence answers Who/What/When/Where, followed by context.",
    task
)

test_prompt(
    "Tweet writer",
    "You write viral tweets. Max 280 chars. Use emojis. Make it exciting. End with relevant hashtags.",
    task
)

print("\n" + "="*60)
print("🧪 EXPERIMENT 2: Zero-Shot vs Few-Shot")
print("="*60)

test_prompt(
    "Zero-shot sentiment",
    "Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.",
    "The food was okay but the service was terrible and we waited 45 minutes."
)

test_prompt(
    "Few-shot sentiment",
    """Classify sentiment. Examples:
"The movie was amazing!" → POSITIVE
"The product broke after 2 days." → NEGATIVE
"The weather is cloudy today." → NEUTRAL

Now classify:""",
    "The food was okay but the service was terrible and we waited 45 minutes."
)

print("\n" + "="*60)
print("🧪 EXPERIMENT 3: Chain-of-Thought")
print("="*60)

test_prompt(
    "Without CoT (direct answer)",
    "Answer math word problems directly with just the number.",
    "A train leaves at 9am going 60km/h. Another leaves at 10am going 90km/h. When does the second train catch up?"
)

test_prompt(
    "With Chain-of-Thought",
    "Solve step by step. Show all calculations. Then give the final answer.",
    "A train leaves at 9am going 60km/h. Another leaves at 10am going 90km/h. When does the second train catch up?"
)