"""
Lab 03 — Agente ReAct baseline con tools y trazas.

El modelo debe responder en JSON:
{"thought": "...", "action": {"name": "execute_query", "args": {"query": "SELECT ..."}}}
o
{"final": "..."}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_REGISTRY

load_dotenv()


SYSTEM_PROMPT = """Eres un agente analitico.
Puedes usar tools para leer datos, calcular estadisticas y generar graficos.
Responde siempre con JSON valido.
Si necesitas usar una tool, responde:
{"thought": "razonamiento breve", "action": {"name": "tool_name", "args": {...}}}
Si ya tienes la respuesta final, responde:
{"final": "respuesta clara para el usuario con supuestos y limitaciones"}
No inventes datos. Si la evidencia no alcanza, dilo.
"""


@dataclass
class TraceStep:
    step: int
    thought: str | None
    action: dict[str, Any] | None
    observation: Any | None


class AnalystAgent:
    def __init__(self, model: str | None = None, max_steps: int = 8):
        self.model = model or os.getenv("AGENT_MODEL") or "REVISAR_MODELO_VIGENTE"
        self.max_steps = max_steps
        self.client = OpenAI()
        self.trace_dir = Path(os.getenv("AGENT_TRACE_DIR", "traces"))
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _call_llm(self, messages: list[dict[str, str]]) -> dict:
        if self.model.startswith("REVISAR_"):
            raise ValueError("Define AGENT_MODEL en .env antes de ejecutar el agente.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def _execute_action(self, action: dict[str, Any]) -> Any:
        name = action.get("name")
        args = action.get("args", {})
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            return {"error": f"Tool no registrada: {name}"}
        return tool(**args)

    def run(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[TraceStep] = []

        for step_number in range(1, self.max_steps + 1):
            model_output = self._call_llm(messages)
            if "final" in model_output:
                result = {
                    "question": question,
                    "answer": model_output["final"],
                    "trace": [asdict(step) for step in trace],
                    "status": "completed",
                }
                self._write_trace(result)
                return result

            action = model_output.get("action")
            thought = model_output.get("thought")
            observation = self._execute_action(action or {})
            trace.append(TraceStep(step_number, thought, action, observation))
            messages.append({"role": "assistant", "content": json.dumps(model_output, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(observation, ensure_ascii=False)}"})

        result = {
            "question": question,
            "answer": "No pude completar la tarea dentro del limite de pasos.",
            "trace": [asdict(step) for step in trace],
            "status": "max_steps_reached",
        }
        self._write_trace(result)
        return result

    def _write_trace(self, result: dict) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.trace_dir / f"trace-{timestamp}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    agent = AnalystAgent()
    question = "REEMPLAZAR por una pregunta analitica sobre tu dataset."
    print(json.dumps(agent.run(question), indent=2, ensure_ascii=False))