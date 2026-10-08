from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from myagent import (
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    MyAnalystAgent,
)

class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    question: str
    answer: str
    status: str
    steps: int
    trace: list[dict[str, Any]]
    model: str
    usage: dict[str, Any]
    
class MyAnalystAgentLangGraph:
    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 8,
    ):
        self.model = (
            model
            or os.getenv("AGENT_MODEL")
            or "REVISAR_MODELO_VIGENTE"
        )

        self.max_steps = max_steps

        # Reutilizamos el agente que ya funciona.
        self.base_agent = MyAnalystAgent(
            model=self.model,
            max_steps=max_steps,
        )

        self.graph = self._build_graph()
        
    def _modelo(self, state: AgentState) -> dict:
        messages = state["messages"]

        model_output, assistant_message, usage_data = (
            self.base_agent._call_llm(messages)
        )

        trace = list(state.get("trace", []))

        trace.append({
            "step": state["steps"] + 1,
            "model_output": model_output,
            "usage": usage_data,
        })

        return {
            "messages": messages + (
                [assistant_message]
                if assistant_message is not None
                else [{
                    "role": "assistant",
                    "content": json.dumps(
                        model_output,
                        ensure_ascii=False,
                    ),
                }]
            ),
            "trace": trace,
            "steps": state["steps"] + 1,
            "usage": usage_data,
        }
        
    def _router(self, state: AgentState) -> str:
        last_trace = state["trace"][-1]
        model_output = last_trace["model_output"]

        if "final" in model_output:
            return "final"

        if state["steps"] >= self.max_steps:
            return "max_steps"

        return "tool"
    
    def _herramienta(self, state: AgentState) -> dict:
        last_trace = state["trace"][-1]
        model_output = last_trace["model_output"]

        action = model_output.get("action")

        observation = self.base_agent._execute_action(
            action or {}
        )

        tool_name = None

        if isinstance(action, dict):
            tool_name = action.get("name")

        model_observation = self.base_agent._observation_for_model(
            tool_name,
            observation,
        )

        messages = list(state["messages"])

        messages.append({
            "role": "user",
            "content": (
                "Observation: "
                + json.dumps(
                    model_observation,
                    ensure_ascii=False,
                )
            ),
        })

        trace = list(state["trace"])

        trace[-1]["action"] = action
        trace[-1]["observation"] = observation

        return {
            "messages": messages,
            "trace": trace,
        }
        
    def _final(self, state: AgentState) -> dict:
        model_output = state["trace"][-1]["model_output"]

        return {
            "answer": model_output["final"],
            "status": "completed",
        }
        
    def _max_steps(self, state: AgentState) -> dict:
        return {
            "answer": "No pude completar la tarea dentro del limite de pasos.",
            "status": "max_steps_reached",
        }
        
    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("modelo", self._modelo)
        graph.add_node("herramienta", self._herramienta)
        graph.add_node("final", self._final)
        graph.add_node("max_steps", self._max_steps)

        graph.add_edge(START, "modelo")

        graph.add_conditional_edges(
            "modelo",
            self._router,
            {
                "tool": "herramienta",
                "final": "final",
                "max_steps": "max_steps",
            },
        )

        graph.add_edge("herramienta", "modelo")

        graph.add_edge("final", END)
        graph.add_edge("max_steps", END)

        return graph.compile()
    
    def run(self, question: str) -> dict:
        initial_state: AgentState = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "question": question,
            "answer": "",
            "status": "running",
            "steps": 0,
            "trace": [],
            "model": self.model,
            "usage": {},
        }

        result = self.graph.invoke(initial_state)

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "trace": result.get("trace", []),
            "status": result.get("status", "unknown"),
            "model": self.model,
            "usage": result.get("usage", {}),
        }
        
if __name__ == "__main__":
    agent = MyAnalystAgentLangGraph()

    result = agent.run(
        "¿Cuánto se vendió en marzo de 2026?"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )