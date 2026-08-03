"""
Week 4 · Day 3 — Applied Practice: Research Workflow Graph
==============================================================
State: query, search_results, draft, feedback, iteration, final_report.

    search -> draft -> critique -> [meets threshold: finalize | else: revise (loop back to draft)]

The critique node scores the draft 0-10. Below `QUALITY_THRESHOLD` sends it
back to `draft` with the feedback attached; at/above the threshold — or once
`MAX_ITERATIONS` is hit, whichever comes first — it moves to `finalize`.
The hard iteration cap is what turns a potentially infinite loop into a
graph that is guaranteed to terminate.

Run:
    python research_workflow.py "How does LangGraph handle interrupts?"
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
QUALITY_THRESHOLD = 8   # out of 10
MAX_ITERATIONS = 3


class _OfflineLLM:
    def invoke(self, messages):
        class _Resp:
            def __init__(self, content):
                self.content = content
        system = messages[0].content
        if "score the draft" in system.lower():
            return _Resp("6/10 — needs one more concrete example.")
        if "you are a researcher" in system.lower():
            return _Resp("[offline draft] LangGraph interrupts pause execution at a node "
                          "boundary and persist state via a checkpointer, so the graph can "
                          "resume from that exact node later.")
        return _Resp("[offline mock search result]")


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        print("[warn] GROQ_API_KEY not set — using deterministic offline stub.\n")
        return _OfflineLLM()
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=0.3)


llm = get_llm()


class ResearchState(TypedDict):
    query: str
    search_results: str
    draft: str
    feedback: str
    iteration: int
    quality_score: int
    final_report: str


def search(state: ResearchState) -> dict:
    """Stand-in for a real retrieval step (web search / vector DB)."""
    prompt = f"Give 3 concise bullet facts relevant to researching: {state['query']}"
    result = llm.invoke([SystemMessage(content="You are a research assistant gathering raw facts."),
                          HumanMessage(content=prompt)])
    print(f"[search] gathered background notes")
    return {"search_results": result.content}


def draft_report(state: ResearchState) -> dict:
    iteration = state["iteration"] + 1
    feedback_note = f"\nPrevious feedback to address: {state['feedback']}" if state["feedback"] else ""
    prompt = (
        f"Query: {state['query']}\n"
        f"Background notes:\n{state['search_results']}{feedback_note}\n\n"
        "Write a concise, accurate 3-sentence report answering the query."
    )
    result = llm.invoke([SystemMessage(content="You are a researcher writing a short factual report."),
                          HumanMessage(content=prompt)])
    print(f"[draft_report] iteration {iteration} draft written")
    return {"draft": result.content, "iteration": iteration}


def critique(state: ResearchState) -> dict:
    prompt = (
        f"Query: {state['query']}\nDraft:\n{state['draft']}\n\n"
        "Score the draft's accuracy and completeness from 0-10 and give one "
        "sentence of feedback. Respond as: '<score>/10 - <feedback>'"
    )
    result = llm.invoke([SystemMessage(content="You are a strict quality reviewer. "
                                                 "Score the draft honestly."),
                          HumanMessage(content=prompt)])
    text = result.content.strip()
    try:
        score = int(text.split("/")[0].strip())
    except (ValueError, IndexError):
        score = 5  # fail-safe: never crash the graph on an unparsable score
    print(f"[critique] score={score}/10 -> {text}")
    return {"quality_score": score, "feedback": text}


def route_after_critique(state: ResearchState) -> str:
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "finalize"
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"[route_after_critique] hit MAX_ITERATIONS={MAX_ITERATIONS}, finalizing anyway")
        return "finalize"
    return "revise"


def finalize(state: ResearchState) -> dict:
    print(f"[finalize] accepted after {state['iteration']} iteration(s), score={state['quality_score']}")
    return {"final_report": state["draft"]}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("search", search)
    graph.add_node("draft", draft_report)
    graph.add_node("critique", critique)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "search")
    graph.add_edge("search", "draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"finalize": "finalize", "revise": "draft"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "How does LangGraph handle interrupts?"
    app = build_graph()

    result = app.invoke({
        "query": query,
        "search_results": "",
        "draft": "",
        "feedback": "",
        "iteration": 0,
        "quality_score": 0,
        "final_report": "",
    })

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result["final_report"])
    print(f"\n(reached in {result['iteration']} iteration(s), final score {result['quality_score']}/10)")


if __name__ == "__main__":
    main()
