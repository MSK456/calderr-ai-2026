"""
Week 4 · Day 1 — LangGraph Foundations
=======================================
Core learning: the State -> Node -> Edge mental model.

This file builds the SIMPLEST possible LangGraph graph on purpose: a linear,
three-node pipeline over a typed state. No branching, no loops, no LLM calls.
The goal is to internalise four concepts before anything gets complicated:

    1. State   — a single TypedDict that every node reads from and writes to.
    2. Node    — a plain Python function: (state) -> partial state update.
    3. Edge    — a directed connection between two nodes (or START/END).
    4. Compile — StateGraph.compile() turns the graph definition into a
                 runnable object with .invoke() / .stream().

Scenario: a support-ticket intake pipeline.
    intake -> classify -> draft_response -> END

Run:
    python langgraph_foundations.py
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. State — the single source of truth passed between every node.
# ---------------------------------------------------------------------------
class TicketState(TypedDict):
    """Everything the pipeline knows about one support ticket."""

    raw_text: str          # what the customer typed
    category: str          # filled in by `classify`
    priority: str           # filled in by `classify`
    draft_response: str    # filled in by `draft_response`


# ---------------------------------------------------------------------------
# 2. Nodes — each one is just a function. LangGraph does not care how the
#    work gets done inside a node, only that it returns a dict of the state
#    keys it wants to update.
# ---------------------------------------------------------------------------
def intake(state: TicketState) -> dict:
    """Normalise the raw ticket text (strip whitespace, enforce non-empty)."""
    text = state["raw_text"].strip()
    if not text:
        text = "(empty ticket — no content provided)"
    print(f"[intake]      normalised text -> {text!r}")
    return {"raw_text": text}


def classify(state: TicketState) -> dict:
    """
    Rule-based classification stand-in.

    In `day2_branching_loops/classification_router.py` this exact step is
    upgraded into an LLM-backed router with conditional edges. Here it stays
    deterministic on purpose, so the *graph mechanics* are the only new idea.
    """
    text = state["raw_text"].lower()

    if any(word in text for word in ("refund", "invoice", "charge", "billing")):
        category, priority = "billing", "high"
    elif any(word in text for word in ("bug", "error", "crash", "not working")):
        category, priority = "technical", "high"
    else:
        category, priority = "general", "normal"

    print(f"[classify]    category={category!r} priority={priority!r}")
    return {"category": category, "priority": priority}


def draft_response(state: TicketState) -> dict:
    """Compose a short acknowledgement based on the classification."""
    templates = {
        "billing": "Thanks for reaching out about your billing concern — "
                    "our finance team will review your account within 24 hours.",
        "technical": "Sorry you're hitting an issue. A support engineer will "
                      "follow up with troubleshooting steps shortly.",
        "general": "Thanks for your message — we'll get back to you soon.",
    }
    draft = templates[state["category"]]
    print(f"[draft_response] {draft!r}")
    return {"draft_response": draft}


# ---------------------------------------------------------------------------
# 3 & 4. Edges + compilation — wire the nodes together in a straight line.
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("intake", intake)
    graph.add_node("classify", classify)
    graph.add_node("draft_response", draft_response)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")
    graph.add_edge("classify", "draft_response")
    graph.add_edge("draft_response", END)

    return graph.compile()


def main() -> None:
    app = build_graph()

    sample_tickets = [
        "I was charged twice for my subscription this month, please refund me.",
        "The app crashes every time I try to upload a photo.",
        "Just wanted to say the new dashboard looks great!",
    ]

    for ticket in sample_tickets:
        print("\n" + "=" * 70)
        print(f"TICKET: {ticket}")
        print("=" * 70)
        result = app.invoke(
            {"raw_text": ticket, "category": "", "priority": "", "draft_response": ""}
        )
        print("-" * 70)
        print(f"FINAL STATE: {result}")


if __name__ == "__main__":
    main()
