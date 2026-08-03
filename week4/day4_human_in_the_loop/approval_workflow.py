"""
Week 4 · Day 4 — Human-in-the-Loop: Interrupt Patterns
==========================================================
Core learning: `interrupt()`, `Command(resume=...)`, and why a checkpointer
is mandatory for HITL (the graph must be able to pause mid-run and be
resumed later — even in a different process — from the exact node it
stopped at).

Flow:
    propose_action -> human_review (INTERRUPTS here) -> apply_decision -> END

How an interrupt actually works, mechanically:
    1. `interrupt(payload)` inside a node raises a special control-flow
       signal. LangGraph catches it, saves a checkpoint of the ENTIRE state
       at that exact point (via the configured checkpointer), and returns
       control to the caller with `__interrupt__` set in the result.
    2. The caller inspects `__interrupt__` (or `app.get_state(config)`) to
       see what's being asked, e.g. "approve this action?".
    3. The caller resumes by invoking the SAME thread_id again with
       `Command(resume=<value>)`. LangGraph loads the checkpoint, injects
       `<value>` as the interrupt()'s return value, and continues execution
       from that exact node — not from the start of the graph.

This is what makes "pause for days, come back, resume" possible: the
checkpoint is durable, so step 3 can happen in a completely new process.

Run:
    python approval_workflow.py
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    action: str
    risk_level: str
    decision: str      # "approved" | "rejected"
    outcome: str


def propose_action(state: ApprovalState) -> dict:
    action = "Delete 90-day-old log files from the archive bucket"
    risk_level = "medium"  # any action touching data deletion is never auto-approved here
    print(f"[propose_action] agent proposes: {action!r} (risk={risk_level})")
    return {"action": action, "risk_level": risk_level}


def human_review(state: ApprovalState) -> dict:
    """
    Pauses the graph and hands control back to a human reviewer.
    The dict passed to `interrupt()` is exactly what a UI (or this CLI) shows
    the human to make their decision.
    """
    decision = interrupt(
        {
            "type": "approval_request",
            "action": state["action"],
            "risk_level": state["risk_level"],
            "instructions": "Reply 'approved' or 'rejected'.",
        }
    )
    print(f"[human_review] received human decision: {decision!r}")
    return {"decision": decision}


def apply_decision(state: ApprovalState) -> dict:
    if state["decision"] == "approved":
        outcome = f"Executed: {state['action']}"
    else:
        outcome = f"Skipped (rejected by reviewer): {state['action']}"
    print(f"[apply_decision] {outcome}")
    return {"outcome": outcome}


def build_graph():
    graph = StateGraph(ApprovalState)
    graph.add_node("propose_action", propose_action)
    graph.add_node("human_review", human_review)
    graph.add_node("apply_decision", apply_decision)

    graph.add_edge(START, "propose_action")
    graph.add_edge("propose_action", "human_review")
    graph.add_edge("human_review", "apply_decision")
    graph.add_edge("apply_decision", END)

    # A checkpointer is REQUIRED for interrupt()/Command(resume=...) to work.
    # InMemorySaver is fine for a demo; production graphs use SqliteSaver
    # (see day5_production_graphs) so the pause can survive a process restart.
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    app = build_graph()
    config = {"configurable": {"thread_id": "approval-demo-1"}}

    print("=" * 70)
    print("STEP 1 — run until the graph pauses for human input")
    print("=" * 70)
    result = app.invoke(
        {"action": "", "risk_level": "", "decision": "", "outcome": ""},
        config=config,
    )
    print(f"\nGraph paused. Interrupt payload:\n  {result['__interrupt__'][0].value}\n")

    print("=" * 70)
    print("STEP 2 — a human types their decision (simulated here)")
    print("=" * 70)
    human_input = input("Approve the proposed action? [approved/rejected]: ").strip().lower()
    if human_input not in ("approved", "rejected"):
        print("(defaulting to 'rejected' — unrecognised input)")
        human_input = "rejected"

    print("\n" + "=" * 70)
    print("STEP 3 — resume the SAME thread_id from the exact interrupted node")
    print("=" * 70)
    final_result = app.invoke(Command(resume=human_input), config=config)
    print(f"\nFINAL STATE: {final_result}")


if __name__ == "__main__":
    main()
