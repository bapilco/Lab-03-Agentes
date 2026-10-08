"""
Lab 03 — Agente analítico implementado con LangGraph.

Esta versión mantiene el contrato de MyAnalystAgent y traslada el bucle
ReAct a un grafo explícito:

    START -> modelo -> router -> herramienta -> router_herramienta -> modelo
                       |                         |
                       +-> responder             +-> tope
                       +-> token                 +-> repeticion
                       +-> error

El grafo vive dentro de la clase MyAnalystAgentLangGraph.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from myagent import MyAnalystAgent
from pathlib import Path


class AgentState(TypedDict, total=False):
    # Historial exacto que se envía al modelo.
    messages: list[dict[str, Any]]

    question: str

    # Salida del modelo en el paso actual.
    model_output: dict[str, Any] | None
    assistant_message: dict[str, Any] | None

    # Acción/observación del paso actual.
    action: dict[str, Any] | None
    observation: Any

    # Estado de ejecución.
    steps: int
    repeat_counts: dict[str, int]

    # Resultado final.
    answer: str
    status: str

    # Evidencia y métricas.
    trace: list[dict[str, Any]]
    usage: dict[str, Any]
    step_usage: dict[str, Any]
    model: str


class MyAnalystAgentLangGraph(MyAnalystAgent):
    """
    Versión LangGraph del agente de la Parte 3.

    Mantiene las herramientas, el prompt, los frenos y el contrato
    del agente original, pero el flujo de control queda representado
    explícitamente mediante nodos y aristas.
    """

    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 8,
        token_budget: int = 10000,
        repeat_limit: int = 3,
    ):
        super().__init__(
            model=model,
            max_steps=max_steps,
            token_budget=token_budget,
            repeat_limit=repeat_limit,
        )

        trace_dir_env = os.getenv("AGENT_TRACE_DIR")

        if trace_dir_env:
            self.trace_dir = Path(trace_dir_env).expanduser().resolve()
        else:
            self.trace_dir = Path(__file__).resolve().parent / "traces"

        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # NODO 1: MODELO
    # ------------------------------------------------------------------
    def _modelo(self, state: AgentState) -> dict[str, Any]:
        """
        Llama al modelo.

        El presupuesto de tokens se revisa ANTES de cada llamada.
        """

        messages = state["messages"]
        usage_total = dict(state["usage"])

        estimated_tokens = self._estimate_tokens(messages)

        if estimated_tokens > self.token_budget:
            usage_total["error"] = (
                f"Presupuesto de tokens alcanzado: "
                f"{estimated_tokens} > {self.token_budget}"
            )

            return {
                "status": "token_budget_reached",
                "answer": (
                    "No pude completar la tarea porque se alcanzó "
                    "el presupuesto de tokens."
                ),
                "usage": usage_total,
            }

        try:
            model_output, assistant_message, usage_data = self._call_llm(
                messages
            )
        except Exception as exc:
            error_message = str(exc)
            usage_total["error"] = error_message

            return {
                "status": "error",
                "answer": "No pude completar la tarea debido a un error.",
                "usage": usage_total,
            }

        usage_total["tokens_entrada"] += usage_data.get(
            "tokens_entrada", 0
        )
        usage_total["tokens_salida"] += usage_data.get(
            "tokens_salida", 0
        )
        usage_total["latencia_segundos"] += usage_data.get(
            "latencia_segundos", 0.0
        )

        if usage_data.get("error"):
            usage_total["error"] = usage_data["error"]

        return {
            "model_output": model_output,
            "assistant_message": assistant_message,
            "usage": usage_total,
            "step_usage": usage_data,
            "status": "running",
        }

    # ------------------------------------------------------------------
    # ROUTER DESPUÉS DEL MODELO
    # ------------------------------------------------------------------
    def _router_modelo(self, state: AgentState) -> str:
        """
        Decide qué camino seguir después del modelo.

        Esto hace explícito el branching del agente:
        - respuesta final
        - error
        - presupuesto agotado
        - herramienta
        """

        status = state.get("status")

        if status == "token_budget_reached":
            return "token"

        if status == "error":
            return "error"

        model_output = state.get("model_output") or {}

        if "final" in model_output:
            return "responder"

        return "herramienta"

    # ------------------------------------------------------------------
    # NODO 2: HERRAMIENTA
    # ------------------------------------------------------------------
    def _herramienta(self, state: AgentState) -> dict[str, Any]:
        """
        Ejecuta la herramienta elegida por el modelo.

        El detector de repetición se mantiene antes de ejecutar
        físicamente la herramienta.
        """

        model_output = state.get("model_output") or {}
        action = model_output.get("action")

        repeat_counts = dict(state.get("repeat_counts", {}))
        usage_total = dict(state["usage"])

        if not isinstance(action, dict):
            observation = {
                "error": (
                    "El modelo no produjo una acción válida "
                    "para ejecutar."
                )
            }

            return {
                "action": action,
                "observation": observation,
                "status": "running",
            }

        action_name = action.get("name")
        action_args = action.get("args", {})

        action_key = (
            f"{action_name}:"
            + json.dumps(
                action_args,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

        repeat_counts[action_key] = repeat_counts.get(action_key, 0) + 1

        # Freno de repetición.
        if repeat_counts[action_key] > self.repeat_limit:
            usage_total["error"] = (
                f"Repetición detectada: {action_name} "
                f"con los mismos argumentos "
                f"{repeat_counts[action_key]} veces."
            )

            return {
                "action": action,
                "repeat_counts": repeat_counts,
                "status": "repetition_detected",
                "answer": (
                    "No pude completar la tarea porque detecté "
                    "una repetición de la misma herramienta."
                ),
                "usage": usage_total,
            }

        observation = self._execute_action(action)

        # Se conserva la misma estructura de traza del agente original.
        trace = list(state.get("trace", []))

        step_usage = state.get("step_usage", {})

        trace.append(
            {
                "step": state.get("steps", 0) + 1,
                "thought": model_output.get("thought"),
                "action": action,
                "observation": observation,
                "model": self.model,
                "input_tokens": step_usage.get("tokens_entrada", 0),
                "output_tokens": step_usage.get("tokens_salida", 0),
                "latency_seconds": step_usage.get("latencia_segundos", 0.0),
                "error": step_usage.get("error"),
            }
        )

        messages = list(state["messages"])
        assistant_message = state.get("assistant_message")

        if assistant_message is not None:
            messages.append(assistant_message)

            tool_call_id = assistant_message["tool_calls"][0]["id"]

            tool_name = action.get("name")
            model_observation = self._observation_for_model(
                tool_name,
                observation,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        model_observation,
                        ensure_ascii=False,
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        model_output,
                        ensure_ascii=False,
                    ),
                }
            )

            tool_name = action.get("name")
            model_observation = self._observation_for_model(
                tool_name,
                observation,
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Observation: "
                        + json.dumps(
                            model_observation,
                            ensure_ascii=False,
                        )
                    ),
                }
            )

        return {
            "messages": messages,
            "action": action,
            "observation": observation,
            "steps": state.get("steps", 0) + 1,
            "repeat_counts": repeat_counts,
            "trace": trace,
            "status": "running",
        }

    # ------------------------------------------------------------------
    # ROUTER DESPUÉS DE LA HERRAMIENTA
    # ------------------------------------------------------------------
    def _router_herramienta(self, state: AgentState) -> str:
        """
        Decide si se vuelve al modelo o se activa un freno.

        El tope propio del taller termina en un nodo que responde,
        en lugar de depender únicamente de GraphRecursionError.
        """

        status = state.get("status")

        if status == "repetition_detected":
            return "repeticion"

        if status == "error":
            return "error"

        if state.get("steps", 0) >= self.max_steps:
            return "tope"

        return "modelo"

    # ------------------------------------------------------------------
    # NODO DE RESPUESTA FINAL
    # ------------------------------------------------------------------
    def _responder(self, state: AgentState) -> dict[str, Any]:
        model_output = state.get("model_output") or {}

        return {
            "answer": model_output.get(
                "final",
                "No puedo responder con los datos disponibles.",
            ),
            "status": "completed",
        }

    # ------------------------------------------------------------------
    # NODO DE TOPE
    # ------------------------------------------------------------------
    def _tope(self, state: AgentState) -> dict[str, Any]:
        return {
            "answer": (
                "No pude completar la tarea dentro del limite de pasos."
            ),
            "status": "max_steps_reached",
        }

    # ------------------------------------------------------------------
    # NODOS DE SALIDA DE LOS FRENOS
    # ------------------------------------------------------------------
    def _token(self, state: AgentState) -> dict[str, Any]:
        return {
            "answer": state.get(
                "answer",
                "No pude completar la tarea porque se alcanzó "
                "el presupuesto de tokens.",
            ),
            "status": "token_budget_reached",
        }

    def _repeticion(self, state: AgentState) -> dict[str, Any]:
        return {
            "answer": state.get(
                "answer",
                "No pude completar la tarea porque detecté "
                "una repetición de la misma herramienta.",
            ),
            "status": "repetition_detected",
        }

    def _error(self, state: AgentState) -> dict[str, Any]:
        return {
            "answer": state.get(
                "answer",
                "No pude completar la tarea debido a un error.",
            ),
            "status": "error",
        }

    # ------------------------------------------------------------------
    # CONSTRUCCIÓN DEL GRAFO
    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("modelo", self._modelo)
        graph.add_node("herramienta", self._herramienta)
        graph.add_node("responder", self._responder)
        graph.add_node("tope", self._tope)
        graph.add_node("token", self._token)
        graph.add_node("repeticion", self._repeticion)
        graph.add_node("error", self._error)

        graph.add_edge(START, "modelo")

        graph.add_conditional_edges(
            "modelo",
            self._router_modelo,
            {
                "herramienta": "herramienta",
                "responder": "responder",
                "token": "token",
                "error": "error",
            },
        )

        graph.add_conditional_edges(
            "herramienta",
            self._router_herramienta,
            {
                "modelo": "modelo",
                "tope": "tope",
                "repeticion": "repeticion",
                "error": "error",
            },
        )

        graph.add_edge("responder", END)
        graph.add_edge("tope", END)
        graph.add_edge("token", END)
        graph.add_edge("repeticion", END)
        graph.add_edge("error", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # CONTRATO PÚBLICO
    # ------------------------------------------------------------------
    def run(self, question: str) -> dict:
        """
        Mantiene el contrato exigido por el taller:

            agent = MyAnalystAgentLangGraph()
            result = agent.run(question)

        Devuelve:
            answer, trace, status, model, usage
        """

        self.tool_data = {}

        initial_state: AgentState = {
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "question": question,
            "model_output": None,
            "assistant_message": None,
            "action": None,
            "observation": None,
            "steps": 0,
            "repeat_counts": {},
            "answer": "",
            "status": "running",
            "trace": [],
            "usage": {
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "latencia_segundos": 0.0,
                "error": None,
            },
            "step_usage": {},
            "model": self.model,
        }

        # El límite de LangGraph funciona como red de seguridad.
        # El freno pedagógico sigue siendo el nodo "tope".
        recursion_limit = max(20, self.max_steps * 2 + 5)

        final_state = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": recursion_limit,
            },
        )

        result = {
            "question": question,
            "answer": final_state.get(
                "answer",
                "No puedo responder con los datos disponibles.",
            ),
            "trace": final_state.get("trace", []),
            "status": final_state.get("status", "error"),
            "model": self.model,
            "usage": final_state.get(
                "usage",
                {
                    "tokens_entrada": 0,
                    "tokens_salida": 0,
                    "latencia_segundos": 0.0,
                    "error": None,
                },
            ),
        }

        self._write_trace_langgraph(result)

        return result

    def _system_prompt(self) -> str:
        # SYSTEM_PROMPT es una constante del agente original.
        from myagent import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def _write_trace_langgraph(self, result: dict) -> None:
        """
        Misma finalidad que _write_trace, pero con microsegundos para
        evitar colisiones si se ejecutan varias corridas rápidamente.
        """

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        output_path = (
            self.trace_dir / f"trace-langgraph-{timestamp}.json"
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
 
    agent = MyAnalystAgentLangGraph(
        max_steps=8,
        token_budget=10000,
        repeat_limit=3
    )

    result = agent.run("¿Cuánto se vendió en marzo de 2026?")

    print(result["status"])
    print(result["answer"])
    print(len(result["trace"]))
    print(result["usage"])


    question = "¿Cuánto se vendió en marzo de 2026?"

    print(
        json.dumps(
            agent.run(question),
            indent=2,
            ensure_ascii=False,
        )
    )
