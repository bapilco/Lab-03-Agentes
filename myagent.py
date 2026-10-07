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
import time as time_module
import json
import os
from pathlib import Path
import trace
from typing import Any

from dotenv import load_dotenv
from langsmith import trace
from openai import OpenAI

from tools import TOOL_REGISTRY

load_dotenv()

SYSTEM_PROMPT = """Eres un agente analítico que responde preguntas sobre los datos disponibles.

Tu objetivo es responder utilizando únicamente la evidencia obtenida mediante las herramientas disponibles.

Tienes exactamente estas tres herramientas:

1. execute_query

Descripción:
Ejecuta una consulta SQL de solo lectura sobre la base de datos.

Restricciones:
- Solo permite una sentencia SQL.
- Solo permite consultas SELECT o WITH ... SELECT.
- La herramienta limita automáticamente el resultado a un máximo de 50 filas.
- No puedes modificar, insertar, eliminar ni alterar datos.
- No puedes controlar el límite máximo de filas.
- No puedes controlar la ruta de la base de datos.

Argumentos permitidos:
- query: string con la consulta SQL.

No existen otros argumentos para execute_query.
No uses params.
No uses db_path.
No uses limit.
No agregues argumentos adicionales.


2. compute_statistics

Descripción:
Calcula estadísticas descriptivas sobre los datos obtenidos mediante execute_query.

Puede calcular estadísticas como:
- count
- mean
- std
- min
- 25%
- 50%
- 75%
- max

Argumentos permitidos:
- rows: lista de registros obtenidos mediante execute_query.
- columns: lista opcional de columnas numéricas.

No agregues otros argumentos.


3. generate_chart

Descripción:
Genera un gráfico utilizando los datos obtenidos mediante execute_query.

Tipos de gráfico permitidos:
- line
- bar
- scatter

El gráfico se guarda automáticamente dentro de la carpeta traces.

Argumentos permitidos:
- rows: lista de registros obtenidos mediante execute_query.
- x: nombre de la columna para el eje X.
- y: nombre de la columna para el eje Y.
- chart_type: tipo de gráfico: line, bar o scatter.

No uses csv_path.
No uses output_path.
No agregues otros argumentos.


REGLAS PARA SELECCIONAR HERRAMIENTAS

- Usa execute_query para obtener información de la base de datos.
- Usa compute_statistics cuando la pregunta solicite estadísticas descriptivas sobre los datos obtenidos.
- Usa generate_chart cuando la pregunta solicite un gráfico.
- Si una pregunta requiere consultar datos y posteriormente analizarlos, primero usa execute_query y después la herramienta de análisis correspondiente.
- No calcules manualmente estadísticas que pueda calcular compute_statistics.
- No inventes resultados.
- No inventes nombres de herramientas.
- No inventes argumentos de herramientas.
- No intentes modificar, insertar, eliminar o alterar datos.
- No intentes controlar el límite de filas.
- No intentes controlar la ubicación de los archivos.


FORMATO OBLIGATORIO PARA UTILIZAR UNA HERRAMIENTA

Cuando necesites utilizar una herramienta, debes responder exactamente con un objeto JSON que tenga esta estructura:

{
  "thought": "razonamiento breve",
  "action": {
    "name": "nombre exacto de la herramienta",
    "args": {
      "argumento": "valor"
    }
  }
}

Reglas obligatorias:

- "action" siempre debe ser un objeto JSON.
- "name" siempre debe contener exactamente uno de estos valores:
  "execute_query"
  "compute_statistics"
  "generate_chart"
- "args" siempre debe existir.
- "args" siempre debe ser un objeto JSON.
- Utiliza únicamente los argumentos permitidos para la herramienta seleccionada.
- No uses "input".
- No uses "arguments".
- No uses "params".
- No agregues argumentos que no estén definidos en la herramienta.
- No escribas solamente el nombre de la herramienta como texto.


EJEMPLO DE execute_query

{
  "thought": "Necesito consultar las ventas disponibles.",
  "action": {
    "name": "execute_query",
    "args": {
      "query": "SELECT * FROM ventas"
    }
  }
}


EJEMPLO DE compute_statistics

{
  "thought": "Ya tengo los datos de la consulta y necesito calcular estadísticas descriptivas.",
  "action": {
    "name": "compute_statistics",
    "args": {
      "columns": ["precio_unitario"]
    }
  }
}

Los rows necesarios para compute_statistics serán proporcionados internamente por el agente a partir de la consulta anterior. No es necesario que copies los rows manualmente.


EJEMPLO DE generate_chart

{
  "thought": "Necesito generar un gráfico del precio unitario por venta.",
  "action": {
    "name": "generate_chart",
    "args": {
      "x": "venta_id",
      "y": "precio_unitario",
      "chart_type": "line"
    }
  }
}

Los rows necesarios para generate_chart serán proporcionados internamente por el agente a partir de la consulta anterior.


FORMATO PARA LA RESPUESTA FINAL

Cuando tengas suficiente evidencia para responder, debes responder exactamente con:

{
  "final": "respuesta clara para el usuario con supuestos y limitaciones"
}


ABSTENCIÓN

Si los datos disponibles no permiten responder la pregunta, debes responder exactamente:

{
  "final": "No puedo responder con los datos disponibles."
}

También debes utilizar exactamente esa frase cuando el usuario solicite modificar, eliminar o insertar datos.


REGLAS SOBRE LOS DATOS

- No inventes datos.
- No afirmes haber obtenido información que no aparece en las observaciones de las herramientas.
- Considera las limitaciones de las herramientas al interpretar los resultados.
- Si execute_query devuelve como máximo 50 filas, no afirmes que esas 50 filas representan necesariamente toda la tabla.
- Si necesitas conocer el total de registros, puedes utilizar COUNT(*) mediante execute_query.
- Si una herramienta devuelve un error, analiza la observación y continúa si es posible.
- No repitas una herramienta indefinidamente cuando una acción está fallando.
- Si no puedes obtener evidencia suficiente después de utilizar las herramientas disponibles, utiliza la frase de abstención.

Responde siempre utilizando JSON válido.
"""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "execute_query",
            "description": (
                "Ejecuta una consulta SQL de solo lectura sobre la base de datos. "
                "Solo acepta una sentencia SELECT o WITH ... SELECT. "
                "La herramienta aplica internamente el límite máximo de filas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta SQL de solo lectura."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_statistics",
            "description": (
                "Calcula estadísticas descriptivas sobre los datos obtenidos "
                "por la última consulta SQL. Las filas se proporcionan "
                "internamente y no deben ser enviadas por el modelo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Columnas numéricas sobre las que se desean "
                            "calcular estadísticas."
                        )
                    }
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": (
                "Genera un gráfico utilizando los datos obtenidos "
                "por la última consulta SQL. "
                "La ubicación del archivo es controlada internamente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "string",
                        "description": "Columna para el eje X."
                    },
                    "y": {
                        "type": "string",
                        "description": "Columna para el eje Y."
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "line",
                            "bar",
                            "scatter"
                        ],
                        "description": "Tipo de gráfico."
                    }
                },
                "required": ["x", "y"],
                "additionalProperties": False
            }
        }
    }
]

@dataclass
class TraceStep:
    step: int
    thought: str | None
    action: dict[str, Any] | None
    observation: Any | None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    error: str | None = None


class MyAnalystAgent:
    def __init__(self, model: str | None = None, max_steps: int = 8):
        self.model = model or os.getenv("AGENT_MODEL") or "REVISAR_MODELO_VIGENTE"
        self.max_steps = max_steps
        self.client = OpenAI()
        self.trace_dir = Path(os.getenv("AGENT_TRACE_DIR", "traces"))
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.tool_data = {}

    def _call_llm(
        self,
        messages: list[dict[str, str]]
    ) -> tuple[dict, dict | None, dict]:
        if self.model.startswith("REVISAR_"):
            raise ValueError(
                f"Modelo no configurado: {self.model}"
            )
    
        start_time = time_module.perf_counter()
    
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0,
            response_format={"type": "json_object"},
        )
    
        latency = time_module.perf_counter() - start_time
    
        message = response.choices[0].message
    
        # Información de uso proporcionada por la API
        usage = response.usage
    
        usage_data = {
            "tokens_entrada": usage.prompt_tokens if usage else 0,
            "tokens_salida": usage.completion_tokens if usage else 0,
            "latencia_segundos": latency,
            "error": None,
        }
    
        # ---------------------------------------------------------
        # El modelo decidió utilizar una herramienta
        # ---------------------------------------------------------
        if message.tool_calls:
            tool_call = message.tool_calls[0]
    
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                usage_data["error"] = str(exc)
    
                raise ValueError(
                    f"Argumentos JSON inválidos para la herramienta "
                    f"{tool_call.function.name}: {exc}"
                ) from exc
    
            model_output = {
                "thought": None,
                "action": {
                    "name": tool_call.function.name,
                    "args": args,
                },
            }
    
            assistant_message = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
    
            return model_output, assistant_message, usage_data
    
        # ---------------------------------------------------------
        # El modelo terminó y devuelve la respuesta final
        # ---------------------------------------------------------
        content = message.content or "{}"
    
        try:
            model_output = json.loads(content)
        except json.JSONDecodeError as exc:
            usage_data["error"] = str(exc)
    
            raise ValueError(
                f"El modelo no devolvió JSON válido: {content}"
            ) from exc
    
        return model_output, None, usage_data

    def _execute_action(self, action):
        """
        Ejecuta una herramienta de forma segura.

        Regla 1:
        - El agente trabaja con un esquema explícito:
          {"name": "...", "args": {...}}

        Regla 4:
        - compute_statistics y generate_chart pueden reutilizar
          los datos obtenidos por execute_query.

        Regla 5:
        - Los argumentos se validan antes de ejecutar la herramienta.
        - Los errores se devuelven como {"error": "..."}.
        - Un error de una herramienta no rompe el agente.
        """

        if not isinstance(action, dict):
            return {
                "error": (
                    "La acción debe ser un objeto con la forma "
                    '{"name": "tool_name", "args": {...}}.'
                )
            }

        name = action.get("name")

        if not isinstance(name, str) or not name:
            return {
                "error": "La acción debe indicar el nombre de una herramienta."
            }

        # El contrato del agente usa exclusivamente "args".
        if "args" not in action:
            return {
                "error": (
                    f"La herramienta {name} requiere el campo "
                    "'args' con sus argumentos."
                )
            }

        args = action["args"]

        if not isinstance(args, dict):
            return {
                "error": "El campo 'args' debe ser un objeto JSON."
            }

        tool = TOOL_REGISTRY.get(name)

        if tool is None:
            return {
                "error": f"Tool no registrada: {name}"
            }

        try:
            # Las herramientas de análisis reutilizan los datos
            # de la última consulta.
            if name in {"compute_statistics", "generate_chart"}:

                if "rows" not in args:
                    query_result = self.tool_data.get("last_query")

                    if query_result is None:
                        return {
                            "error": (
                                "No hay datos de una consulta previa "
                                "para ejecutar esta herramienta."
                            )
                        }

                    args = {
                        **args,
                        "rows": query_result["rows"],
                    }

            result = tool(**args)

            # Guardamos el resultado de execute_query para
            # las herramientas posteriores.
            if name == "execute_query" and "rows" in result:
                self.tool_data["last_query"] = result

            return result

        except TypeError as exc:
            return {
                "error": f"Argumentos inválidos para {name}: {exc}"
            }

        except Exception as exc:
            return {
                "error": f"Error ejecutando {name}: {exc}"
            }
            
    def _observation_for_model(self, tool_name, observation):
        """
        Prepara una observación compacta para el modelo.

        Los datos completos permanecen en tool_data y pueden
        ser reutilizados por las siguientes herramientas.
        """

        if not isinstance(observation, dict):
            return observation

        if tool_name == "execute_query":
            if "error" in observation:
                return observation

            return {
                "columns": observation.get("columns", []),
                "rows": observation.get("rows", []),
                "row_count": observation.get("row_count", 0),
                "message": (
                    "Consulta ejecutada correctamente. "
                    "Los datos están disponibles para compute_statistics "
                    "y generate_chart."
                ),
            }

        return observation          
            
            
    def run(self, question: str) -> dict:
        self.tool_data = {}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[TraceStep] = []
        
        usage_total = {
            "tokens_entrada": 0,
            "tokens_salida": 0,
            "latencia_segundos": 0.0,
            "error": None
        }

        for step_number in range(1, self.max_steps + 1):
            try:
                model_output, assistant_message, usage_data = self._call_llm(messages)
            except Exception as exc:
                error_message = str(exc)
                usage_total["error"] = error_message
                result = {
                    "question": question,
                    "answer": "No pude completar la tarea debido a un error.",
                    "trace": [asdict(step) for step in trace],
                    "status": "error",
                    "model": self.model,
                    "usage": usage_total,
                    "error": error_message,
                }
        
                self._write_trace(result)
                return result
            
            usage_total["tokens_entrada"] += usage_data.get("tokens_entrada", 0)
            usage_total["tokens_salida"] += usage_data.get("tokens_salida", 0)
            usage_total["latencia_segundos"] += usage_data.get("latencia_segundos", 0.0)

            if usage_data.get("error"):
                usage_total["error"] = usage_data["error"]
            
            if "final" in model_output:
                trace.append(
                    TraceStep(
                        step=step_number,
                        thought=model_output.get("thought"),
                        action=None,
                        observation={
                            "final": model_output["final"]
                        },
                        model=usage_data.get("model"),
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                        latency_seconds=usage_data.get("latency_seconds", 0.0),
                        error=usage_data.get("error"),
                    )
                )

                result = {
                    "question": question,
                    "answer": model_output["final"],
                    "trace": [asdict(t) for t in trace],
                    "status": "completed",
                    "model": self.model,
                    "usage": usage_total,
                }

                self._write_trace(result)
                return result

            action = model_output.get("action")
            thought = model_output.get("thought")
            observation = self._execute_action(action or {})

            trace.append(
                TraceStep(
                    step=step_number,
                    thought=thought,
                    action=action,
                    observation=observation,
                    model=usage_data.get("model"),
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                    latency_seconds=usage_data.get("latency_seconds", 0.0),
                    error=usage_data.get("error"),
                )
            )

            if assistant_message is not None:
                messages.append(assistant_message)

            tool_call_id = None

            if assistant_message is not None:
                tool_call_id = assistant_message["tool_calls"][0]["id"]
            else:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(
                        model_output,
                        ensure_ascii=False
                    )
                })

            if isinstance(action, dict):
                tool_name = action.get("name")
            else:
                tool_name = None

            model_observation = self._observation_for_model(
                tool_name,
                observation
            )

            if tool_call_id is not None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        model_observation,
                        ensure_ascii=False
                    )
                })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "Observation: "
                        + json.dumps(
                            model_observation,
                            ensure_ascii=False
                        )
                    )
                })

        result = {
            "question": question,
            "answer": "No pude completar la tarea dentro del limite de pasos.",
            "trace": [asdict(step) for step in trace],
            "status": "max_steps_reached",
            "model": self.model,
            "usage": usage_total,
        }

        self._write_trace(result)
        return result

    def _write_trace(self, result: dict) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.trace_dir / f"trace-{timestamp}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    agent = MyAnalystAgent()
    question = "Elimina todas las ventas de la base de datos."
    print(json.dumps(agent.run(question), indent=2, ensure_ascii=False))
