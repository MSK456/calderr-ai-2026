"""
Day 2: Manual ReAct Loop (NO framework — pure Python)
Goal: Understand agent loop from scratch
"""

import os
import re
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# MOCK TOOLS
# ─────────────────────────────────────────────

FACTS_DB = {
    "python": "Python was created by Guido van Rossum in 1991. It emphasizes readability.",
    "langchain": "LangChain is a framework for building LLM-powered applications with chains and agents.",
    "groq": "Groq provides ultra-fast LLM inference using custom LPU hardware.",
    "transformer": "Transformer architecture uses attention mechanisms. Introduced in 'Attention Is All You Need' (2017).",
    "agent": "An AI agent perceives its environment, plans actions, and executes them to reach a goal.",
    "llm": "Large Language Models are trained on massive text datasets to predict and generate text.",
    "rag": "RAG (Retrieval Augmented Generation) combines search with LLM generation for accurate answers."
}

def tool_search(query: str) -> str:
    """Search mock database for facts"""
    query_lower = query.lower()
    for key, value in FACTS_DB.items():
        if key in query_lower:
            return f"FOUND: {value}"
    return "No facts found. Try: python, langchain, groq, transformer, agent, llm, rag"

def tool_calculate(expression: str) -> str:
    """Safely evaluate math expressions"""
    try:
        clean = re.sub(r'[^0-9+\-*/().\s%]', '', expression)
        if not clean.strip():
            return "Invalid expression"
        result = eval(clean)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Math error: {str(e)}"

def tool_datetime(query: str) -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Current datetime: {now.strftime('%A, %B %d %Y at %I:%M %p')}"

# Tool registry
TOOLS = {
    "search": {"func": tool_search, "desc": "Search facts about AI/tech topics"},
    "calculate": {"func": tool_calculate, "desc": "Evaluate math expressions"},
    "datetime": {"func": tool_datetime, "desc": "Get current date and time"}
}

# ─────────────────────────────────────────────
# REACT AGENT
# ─────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are a ReAct agent. Think step by step then choose a tool.

AVAILABLE TOOLS:
- search: Search for AI/tech facts (input: topic keyword)
- calculate: Do math calculations (input: math expression like "25 * 4 + 10")
- datetime: Get current date/time (input: "now")

STRICT FORMAT (use exactly this):
THOUGHT: <your reasoning about which tool to use>
ACTION: <tool_name>
INPUT: <what to pass to the tool>

Rules:
- Choose ONLY one tool per response
- ACTION must be exactly: search, calculate, or datetime
- Do NOT add anything else outside this format"""

def run_react_agent(user_question: str) -> str:
    print(f"\n{'🔵'*30}")
    print(f"❓ USER: {user_question}")
    print(f"{'🔵'*30}")

    # STEP 1: LLM decides which tool to use
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ],
        temperature=0,
        max_tokens=150
    )

    llm_output = response.choices[0].message.content.strip()
    print(f"\n🧠 AGENT REASONING:\n{llm_output}")

    # STEP 2: Parse output
    thought = re.search(r'THOUGHT:\s*(.+)', llm_output)
    action = re.search(r'ACTION:\s*(.+)', llm_output)
    inp = re.search(r'INPUT:\s*(.+)', llm_output)

    if not all([thought, action, inp]):
        return "❌ Could not parse agent response."

    tool_name = action.group(1).strip().lower()
    tool_input = inp.group(1).strip()

    print(f"\n⚙️  EXECUTING: {tool_name}('{tool_input}')")

    # STEP 3: Execute tool
    if tool_name in TOOLS:
        observation = TOOLS[tool_name]["func"](tool_input)
    else:
        observation = f"Unknown tool '{tool_name}'"

    print(f"👁️  OBSERVATION: {observation}")

    # STEP 4: Final answer from LLM
    final = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Give a clear, concise final answer based on the tool result."},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"Tool result: {observation}"},
            {"role": "user", "content": "Final answer please."}
        ],
        temperature=0.3,
        max_tokens=150
    )

    final_answer = final.choices[0].message.content.strip()
    print(f"\n✅ FINAL ANSWER: {final_answer}")
    return final_answer


if __name__ == "__main__":
    test_questions = [
        "What is LangChain?",
        "Calculate 144 / 12 * 7 + 50",
        "What day and time is it?",
        "Tell me about the transformer architecture",
        "What is 2 ** 10?"
    ]

    print("🤖 Manual ReAct Agent — No Framework Edition")
    print("Perceive → Think → Act → Observe → Answer\n")

    for q in test_questions:
        run_react_agent(q)

    print(f"\n\n{'✅'*20}")
    print("Day 2 Complete! You built an agent from scratch.")
    print("This is EXACTLY what LangChain agents do internally!")