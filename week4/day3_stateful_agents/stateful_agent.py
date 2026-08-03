"""
Week 4 · Day 3 — Stateful Agents: TypedDict + Annotated Reducers
====================================================================
Core learning: how state ACCUMULATES across nodes/turns instead of being
overwritten, using `Annotated[type, reducer]`.

By default, when a node returns {"key": value}, LangGraph *replaces*
state["key"] with `value`. That's fine for scalars (a category, a flag) but
wrong for things that should grow over time — a message history, a running
list of tool calls, a log of intermediate results.

`Annotated[list[X], some_reducer_fn]` tells LangGraph: "when a node returns a
value for this key, don't replace the old value — combine it with the new
one using `some_reducer_fn`."

This demo builds a tiny multi-turn agent whose state accumulates three
different things, each with its own reducer:

    messages             -> langgraph's built-in `add_messages` reducer
    tool_calls_log       -> `operator.add` (simple list concatenation)
    intermediate_results -> a custom reducer that de-duplicates by key

Run:
    python stateful_agent.py
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Custom reducer: merge dicts of intermediate results, newer value wins per key.
# ---------------------------------------------------------------------------
def merge_results(current: dict, update: dict) -> dict:
    merged = dict(current or {})
    merged.update(update or {})
    return merged


class AgentState(TypedDict):
    # Built-in reducer: appends new messages to the running conversation.
    messages: Annotated[list[AnyMessage], add_messages]
    # operator.add on lists == list concatenation (['a'] + ['b'] -> ['a','b'])
    tool_calls_log: Annotated[list[str], operator.add]
    # Custom reducer: dict merge instead of list append.
    intermediate_results: Annotated[dict, merge_results]
    turn: int


# ---------------------------------------------------------------------------
# Nodes — notice every node returns ONLY the delta, never the full history.
# The reducers are responsible for combining it with what's already there.
# ---------------------------------------------------------------------------
def receive_user_turn(state: AgentState) -> dict:
    turn = state["turn"] + 1
    print(f"\n--- turn {turn} ---")
    return {"turn": turn}


def simulate_tool_call(state: AgentState) -> dict:
    """Pretend to call a tool and log it — the log accumulates across turns."""
    last_user_msg = state["messages"][-1].content
    tool_name = "kb_lookup" if "?" in last_user_msg else "no_tool_needed"
    print(f"[simulate_tool_call] logging call to: {tool_name!r}")
    return {"tool_calls_log": [tool_name]}


def record_intermediate_result(state: AgentState) -> dict:
    key = f"turn_{state['turn']}_tool"
    value = state["tool_calls_log"][-1]
    print(f"[record_intermediate_result] {key} = {value!r}")
    return {"intermediate_results": {key: value}}


def respond(state: AgentState) -> dict:
    reply = f"(turn {state['turn']}) Noted — {len(state['tool_calls_log'])} tool call(s) so far."
    print(f"[respond] {reply}")
    return {"messages": [AIMessage(content=reply)]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("receive_user_turn", receive_user_turn)
    graph.add_node("simulate_tool_call", simulate_tool_call)
    graph.add_node("record_intermediate_result", record_intermediate_result)
    graph.add_node("respond", respond)

    graph.add_edge(START, "receive_user_turn")
    graph.add_edge("receive_user_turn", "simulate_tool_call")
    graph.add_edge("simulate_tool_call", "record_intermediate_result")
    graph.add_edge("record_intermediate_result", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def main() -> None:
    app = build_graph()

    state: AgentState = {
        "messages": [],
        "tool_calls_log": [],
        "intermediate_results": {},
        "turn": 0,
    }

    user_turns = [
        "What's your refund policy?",
        "Great, thanks!",
        "One more question — how long does a refund take?",
    ]

    for turn_text in user_turns:
        # Feeding messages in via state and letting `add_messages` merge them
        # is exactly how a real multi-turn LangGraph agent keeps history.
        state = app.invoke(
            {**state, "messages": state["messages"] + [HumanMessage(content=turn_text)]}
        )

    print("\n" + "=" * 70)
    print("FINAL ACCUMULATED STATE")
    print("=" * 70)
    print(f"messages ({len(state['messages'])} total):")
    for m in state["messages"]:
        print(f"   [{m.type}] {m.content}")
    print(f"tool_calls_log: {state['tool_calls_log']}")
    print(f"intermediate_results: {state['intermediate_results']}")


if __name__ == "__main__":
    main()
