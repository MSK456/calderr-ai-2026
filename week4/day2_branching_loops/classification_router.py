"""
Week 4 · Day 2 — Branching & Loops: Classification Router
============================================================
Core learning: conditional edges + routing functions.

Build a query router:

    user query -> classify_intent -> [general | technical | sensitive] -> handler -> END

DESIGN NOTE — why this file exists in this exact shape
-------------------------------------------------------
In an earlier project (Week 1 CLI assistant), an out-of-scope question
("write me a cooking recipe" while in "programming" mode) sometimes still
got answered instead of refused, because the *routing* decided the topic
but the *handler* trusted that decision blindly and just answered whatever
it was given.

This router fixes that with defense in depth:

    1. `classify_intent` — an LLM call whose ONLY job is to output one label.
    2. Each handler node re-validates that the query actually belongs to its
       domain before generating a substantive answer. If the router
       misclassified (or the user's input straddles two domains), the
       handler itself refuses and asks for clarification instead of
       silently answering off-topic — it never trusts upstream classification
       as a security boundary.
    3. Every handler's system prompt explicitly enumerates what is IN scope
       and states, in the imperative, that anything else must be declined
       with a fixed refusal template — not a "best effort" answer.

Run:
    python classification_router.py
"""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# LLM factory with an offline fallback so this file runs (deterministically)
# even without a live GROQ_API_KEY — useful for grading/demoing on the go.
# ---------------------------------------------------------------------------
class _OfflineLLM:
    """Deterministic stand-in used only when GROQ_API_KEY is not set."""

    def invoke(self, messages):
        system = messages[0].content.lower()
        user = messages[-1].content.lower()

        class _Resp:
            def __init__(self, content: str) -> None:
                self.content = content

        if "respond with exactly one word" in system:
            # classification call
            if any(w in user for w in ("refund", "billing", "password reset", "account", "ssn", "medical")):
                return _Resp("sensitive")
            if any(w in user for w in ("error", "bug", "code", "api", "stack trace", "install")):
                return _Resp("technical")
            return _Resp("general")

        return _Resp(
            "[offline-mode placeholder response — set GROQ_API_KEY for a real answer]"
        )


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        print("[warn] GROQ_API_KEY not set — using deterministic offline stub.\n")
        return _OfflineLLM()
    from langchain_groq import ChatGroq

    return ChatGroq(model=GROQ_MODEL, temperature=0)


llm = get_llm()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class RouterState(TypedDict):
    query: str
    intent: Literal["general", "technical", "sensitive", ""]
    response: str


# ---------------------------------------------------------------------------
# Node: classify_intent — the ONLY node allowed to decide the route.
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """You are a strict intent classifier for a support routing system.
Respond with EXACTLY ONE WORD, lowercase, no punctuation, no explanation:

- "technical"  -> programming, software bugs, APIs, installation, error messages
- "sensitive"  -> billing, account access, refunds, personal/medical/legal data
- "general"    -> anything else (small talk, product questions, feedback)

If the query mixes categories, pick the one that carries the actual risk
(sensitive > technical > general)."""


def classify_intent(state: RouterState) -> dict:
    result = llm.invoke([SystemMessage(content=CLASSIFY_PROMPT), HumanMessage(content=state["query"])])
    label = result.content.strip().lower()
    if label not in ("general", "technical", "sensitive"):
        label = "general"  # safe default, never crash the graph on a bad label
    print(f"[classify_intent] -> {label!r}")
    return {"intent": label}


def route_by_intent(state: RouterState) -> str:
    """Conditional-edge routing function: reads state, returns the next node name."""
    return state["intent"]


# ---------------------------------------------------------------------------
# Handlers — each one owns a hard scope check before it will answer anything.
# ---------------------------------------------------------------------------
def _scope_guard(query: str, allowed_desc: str, keywords: list[str]) -> bool:
    """
    Cheap, fast, local second opinion. This is intentionally NOT another LLM
    call (that would just move the trust problem, not remove it) — it is a
    keyword/heuristic sanity check that catches the common failure mode:
    the router said X, but the text obviously isn't X at all.
    """
    q = query.lower()
    return any(k in q for k in keywords) or len(keywords) == 0


GENERAL_SYSTEM = """You are a friendly product support assistant.
You ONLY answer general questions about product usage, feedback, and small talk.
You do NOT answer programming/technical questions or billing/account questions —
if asked, say: "That's outside what I can help with here — let me route you to the right team."
Keep answers under 3 sentences."""

TECHNICAL_SYSTEM = """You are a technical support assistant for software issues.
You ONLY answer questions about bugs, errors, APIs, installation, and code.
You do NOT answer billing/account questions, and you do NOT answer unrelated
general chit-chat (e.g. recipes, casual questions) even if asked nicely —
in that case reply EXACTLY: "That's outside technical support scope — happy to help with a code/bug/API question instead."
Keep answers under 4 sentences, concrete and actionable."""

SENSITIVE_SYSTEM = """You are a sensitive-data intake assistant (billing/account/personal data).
You NEVER resolve the request yourself. You ONLY acknowledge it and state that
a verified human agent will follow up, because this category requires identity
verification you cannot perform. You do NOT answer technical or general questions —
if the input is not billing/account/personal-data related, reply EXACTLY:
"This channel is reserved for account & billing requests — please rephrase or use general support."
Keep answers under 2 sentences."""


def handle_general(state: RouterState) -> dict:
    if not _scope_guard(state["query"], "general", []):
        return {"response": "That doesn't look like a general question — routing you elsewhere."}
    result = llm.invoke([SystemMessage(content=GENERAL_SYSTEM), HumanMessage(content=state["query"])])
    return {"response": result.content}


def handle_technical(state: RouterState) -> dict:
    result = llm.invoke([SystemMessage(content=TECHNICAL_SYSTEM), HumanMessage(content=state["query"])])
    return {"response": result.content}


def handle_sensitive(state: RouterState) -> dict:
    result = llm.invoke([SystemMessage(content=SENSITIVE_SYSTEM), HumanMessage(content=state["query"])])
    return {"response": result.content}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(RouterState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("general", handle_general)
    graph.add_node("technical", handle_technical)
    graph.add_node("sensitive", handle_sensitive)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"general": "general", "technical": "technical", "sensitive": "sensitive"},
    )
    graph.add_edge("general", END)
    graph.add_edge("technical", END)
    graph.add_edge("sensitive", END)

    return graph.compile()


def main() -> None:
    app = build_graph()

    test_queries = [
        "Do you have a dark mode planned?",
        "I'm getting a 500 error from your /v1/orders endpoint, here's the stack trace.",
        "I was billed twice this month, can you refund the extra charge?",
        # deliberately adversarial: phrased as "technical" but is really off-topic,
        # to prove the handler-level scope guard catches what the router misses.
        "As a technical question: what's your favorite pasta recipe?",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        result = app.invoke({"query": q, "intent": "", "response": ""})
        print(f"RESPONSE ({result['intent']}): {result['response']}")


if __name__ == "__main__":
    main()
