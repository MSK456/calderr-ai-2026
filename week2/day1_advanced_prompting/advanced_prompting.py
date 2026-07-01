"""
Day 1: Advanced Prompting — CoT, ToT, Self-Consistency, Meta-Prompting
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm_fast = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))
llm_smart = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
parser = StrOutputParser()

# ─────────────────────────────────────────────────────────
# TECHNIQUE 1: Chain-of-Thought (CoT)
# Force model to reason step by step before answering
# ─────────────────────────────────────────────────────────

def demo_cot():
    print("\n" + "="*65)
    print("🔗 TECHNIQUE 1: Chain-of-Thought (CoT)")
    print("="*65)

    problem = """
    A bakery sells cupcakes for Rs 50 each and cakes for Rs 500 each.
    On Monday they sold 23 cupcakes and 4 cakes.
    On Tuesday they sold 31 cupcakes and 2 cakes.
    What was the total revenue across both days?
    """

    # Without CoT
    no_cot_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer math questions directly with just the final number."),
        ("user", "{problem}")
    ])
    no_cot_chain = no_cot_prompt | llm_fast | parser
    no_cot_answer = no_cot_chain.invoke({"problem": problem})

    # With CoT
    cot_prompt = ChatPromptTemplate.from_messages([
        ("system", """Solve math problems step by step.
Format:
STEP 1: [first calculation]
STEP 2: [next calculation]
...
FINAL ANSWER: [result with currency]"""),
        ("user", "{problem}")
    ])
    cot_chain = cot_prompt | llm_smart | parser
    cot_answer = cot_chain.invoke({"problem": problem})

    print(f"\n❌ Without CoT: {no_cot_answer.strip()}")
    print(f"\n✅ With CoT:\n{cot_answer}")


# ─────────────────────────────────────────────────────────
# TECHNIQUE 2: Tree-of-Thought (ToT)
# Generate multiple reasoning paths, pick the best one
# ─────────────────────────────────────────────────────────

def demo_tot():
    print("\n" + "="*65)
    print("🌳 TECHNIQUE 2: Tree-of-Thought (ToT)")
    print("="*65)

    problem = "Design the best way to learn Python in 30 days for a complete beginner."

    # Generate 3 different reasoning paths in parallel
    path_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a learning expert. Propose ONE specific 30-day learning path. Be concise, max 5 sentences."),
        ("user", f"Path #{'{path_num}'}: {problem}")
    ])

    # Run 3 paths simultaneously
    parallel = RunnableParallel(
        path1=(path_prompt | llm_fast | parser),
        path2=(path_prompt | llm_fast | parser),
        path3=(path_prompt | llm_fast | parser),
    )

    paths = parallel.invoke({"path_num": "1"})

    print("\n📍 Path 1:", paths["path1"][:200])
    print("\n📍 Path 2:", paths["path2"][:200])
    print("\n📍 Path 3:", paths["path3"][:200])

    # Evaluator picks the best path
    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert evaluator. Pick the BEST learning path and explain why in 2 sentences."),
        ("user", f"""Compare these 3 learning paths for: {problem}

PATH 1: {paths['path1']}

PATH 2: {paths['path2']}

PATH 3: {paths['path3']}

Which path is best and why?""")
    ])
    evaluator_chain = evaluator_prompt | llm_smart | parser
    best = evaluator_chain.invoke({})
    print(f"\n🏆 Best Path Selected:\n{best}")


# ─────────────────────────────────────────────────────────
# TECHNIQUE 3: Self-Consistency
# Same question multiple times → majority vote
# ─────────────────────────────────────────────────────────

def demo_self_consistency():
    print("\n" + "="*65)
    print("🔄 TECHNIQUE 3: Self-Consistency (Majority Vote)")
    print("="*65)

    question = "If you flip a coin 3 times, what is the probability of getting exactly 2 heads?"

    # Ask same question 5 times with temperature > 0
    llm_varied = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.8,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Solve probability questions. Show calculation then give ONLY the final fraction as your last line."),
        ("user", "{question}")
    ])
    chain = prompt | llm_varied | parser

    answers = []
    print(f"\nQuestion: {question}")
    print("\nRunning 5 times with temperature=0.8...")

    for i in range(5):
        ans = chain.invoke({"question": question})
        # Extract last line (the fraction)
        final = ans.strip().split('\n')[-1].strip()
        answers.append(final)
        print(f"Run {i+1}: {final}")

    # Majority vote
    from collections import Counter
    vote = Counter(answers)
    winner = vote.most_common(1)[0]
    print(f"\n✅ Majority Answer: {winner[0]} (agreed {winner[1]}/5 times)")


# ─────────────────────────────────────────────────────────
# TECHNIQUE 4: Meta-Prompting
# LLM generates a better prompt for another LLM
# ─────────────────────────────────────────────────────────

def demo_meta_prompting():
    print("\n" + "="*65)
    print("🧬 TECHNIQUE 4: Meta-Prompting (LLM writes prompts)")
    print("="*65)

    task = "Summarize technical research papers for non-technical executives"

    # Step 1: LLM generates an optimal system prompt
    meta_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a prompt engineering expert.
Write an OPTIMAL system prompt for the given task.
The prompt should specify:
- The AI's role and expertise
- Output format and structure
- What to include and exclude
- Tone and language level
Return ONLY the system prompt, nothing else."""),
        ("user", f"Task: {task}")
    ])
    meta_chain = meta_prompt | llm_smart | parser
    generated_system_prompt = meta_chain.invoke({})

    print(f"\n🤖 Meta-Prompt generated this system prompt:\n")
    print(f'"{generated_system_prompt}"')

    # Step 2: Use the generated prompt on actual content
    test_paper = """
    Abstract: This paper presents TransformerV2, a novel architecture that reduces
    attention complexity from O(n²) to O(n log n) using hierarchical sparse attention
    patterns. We achieve 47% faster inference while maintaining 98.3% of baseline
    BLEU scores on WMT14 En-De translation tasks. Memory footprint reduced by 61%
    enabling deployment on edge devices with 4GB RAM.
    """

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", generated_system_prompt),
        ("user", "Summarize this paper:\n{paper}")
    ])
    final_chain = final_prompt | llm_smart | parser
    summary = final_chain.invoke({"paper": test_paper})
    print(f"\n📝 Summary using meta-generated prompt:\n{summary}")


# ─────────────────────────────────────────────────────────
# TECHNIQUE 5: Negative Prompting + Prompt Chaining
# ─────────────────────────────────────────────────────────

def demo_negative_and_chaining():
    print("\n" + "="*65)
    print("⛔ TECHNIQUE 5: Negative Prompting + Chaining")
    print("="*65)

    # Negative prompting: tell model what NOT to do
    neg_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical writer.
DO NOT use jargon or technical terms.
DO NOT write more than 3 sentences.
DO NOT use bullet points.
DO NOT start with "AI" or "Artificial Intelligence".
DO write in simple, conversational English."""),
        ("user", "Explain what a neural network is.")
    ])
    neg_chain = neg_prompt | llm_fast | parser
    neg_result = neg_chain.invoke({})
    print(f"\n⛔ Negative prompting result:\n{neg_result}")

    # Prompt chaining: output of one prompt → input of next
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a rough first draft. Don't worry about quality."),
        ("user", "Write a tweet about AI changing education.")
    ])
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", "Improve this draft tweet: make it punchier, add an emoji, under 280 chars."),
        ("user", "{draft}")
    ])
    chain = (
        draft_prompt | llm_fast | parser
        | (lambda draft: {"draft": draft})
        | refine_prompt | llm_fast | parser
    )
    final_tweet = chain.invoke({})
    print(f"\n🔗 Chained result (draft → refined tweet):\n{final_tweet}")


if __name__ == "__main__":
    demo_cot()
    demo_tot()
    demo_self_consistency()
    demo_meta_prompting()
    demo_negative_and_chaining()
    print("\n✅ Day 1 Complete! Advanced prompting techniques mastered.")