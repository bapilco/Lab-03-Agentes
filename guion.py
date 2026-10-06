"""
Lab 03 — Un «modelo» de guion para la Parte 0 del Taller 3. No llama a ningún LLM.

    python guion.py ataque     # 0.b · la consulta que pasa el guard, y la corrida que muere
    python guion.py limite     # 0.c · el límite que viaja en el esquema público

Sustituye al LLM por una lista fija de salidas —las mismas que `agent.py` espera:
`{"thought": ..., "action": {...}}` o `{"final": ...}`— y corre el `AnalystAgent` del
andamiaje **sin tocarlo**. Así las dos fallas se ven igual en cualquier máquina, sin clave
y sin VPN, y lo que se observa es el andamiaje, no el humor de un modelo.

Lo que este archivo NO cambia es la parte interesante: el bucle, `_execute_action`, las
tres herramientas y la escritura de la traza son los de `agent.py` y `tools.py` tal como se
reparten. Si una de las dos fallas no aparece, cambiaste el andamiaje y tienes que decir
qué.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

from agent import AnalystAgent

DB = Path(os.getenv("AGENT_DB_PATH", "data/database.sqlite"))


class AgenteDeGuion(AnalystAgent):
    """El agente del laboratorio con el modelo sustituido por un guion fijo."""

    def __init__(self, guion: list[dict], max_steps: int = 8):
        # No se llama al __init__ original: construiría un cliente de OpenAI y pediría
        # clave. Todo lo demás —el bucle, las herramientas, la traza— es el del andamiaje.
        self.model = "guion"
        self.max_steps = max_steps
        self.trace_dir = Path(os.getenv("AGENT_TRACE_DIR", "traces"))
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._guion = list(guion)

    def _call_llm(self, messages):
        return self._guion.pop(0) if self._guion else {"final": "fin del guion"}


def _accion(nombre: str, **args) -> dict:
    return {"thought": f"llamo a {nombre}", "action": {"name": nombre, "args": args}}


def _trazas(agente: AnalystAgent) -> int:
    return len(list(agente.trace_dir.glob("trace-*.json")))


def ataque() -> None:
    """0.b — Una consulta que el guard deja pasar, y lo que le hace a la corrida."""
    if not DB.exists():
        sys.exit("Primero: python crear_base.py")
    consulta = "select 1;drop table ventas"
    agente = AgenteDeGuion([_accion("execute_query", query=consulta)])
    antes = _trazas(agente)
    print(f"consulta del «modelo»: {consulta!r}")
    try:
        agente.run("¿Cuántas ventas hay?")
        print("la corrida terminó")
    except Exception as e:  # el andamiaje no la captura: aquí se ve por qué importa
        print(f"la corrida MURIÓ: {type(e).__name__}: {str(e)[:160]}")
    print(f"trazas escritas en {agente.trace_dir}/: {antes} antes, {_trazas(agente)} después")
    with sqlite3.connect(DB) as conn:
        n = conn.execute("SELECT COUNT(*) FROM ventas").fetchone()[0]
    print(f"la tabla ventas sigue ahí: {n} filas")


def limite() -> None:
    """0.c — `row_limit` y `output_path` los decide quien llama a la herramienta."""
    if not DB.exists():
        sys.exit("Primero: python crear_base.py")
    fuera = Path("..") / "fuera_del_lab" / "grafico.png"
    guion = [
        _accion("execute_query", query="SELECT * FROM ventas LIMIT 10000"),
        _accion("generate_chart", csv_path="data/ventas_mensuales.csv", x="mes",
                y="importe", output_path=str(fuera)),
        _accion("generate_chart", csv_path="data/ventas_mensuales.csv", x="mes",
                y="n_ventas"),
        _accion("generate_chart", csv_path="data/ventas_mensuales.csv", x="mes",
                y="importe", chart_type="bar"),
        {"final": "listo"},
    ]
    agente = AgenteDeGuion(guion)
    resultado = agente.run("Dame todas las ventas y dos gráficos.")
    obs = [paso["observation"] for paso in resultado["trace"]]
    filas = obs[0].get("row_count")
    print(f"1 · LIMIT 10000 pasó el guard: {filas} filas devueltas "
          f"(el tope por defecto es 50) · {len(json.dumps(obs[0], ensure_ascii=False)):,} "
          f"caracteres de observación que vuelven al modelo en cada paso siguiente")
    print(f"2 · el modelo eligió la ruta, fuera del laboratorio: {fuera} existe = {fuera.exists()}")
    rutas = [o.get("chart_path") for o in obs[2:4]]
    print(f"3 · dos gráficos distintos, una sola ruta: {rutas} — el segundo pisó al primero")


if __name__ == "__main__":
    casos = {"ataque": ataque, "limite": limite}
    if len(sys.argv) != 2 or sys.argv[1] not in casos:
        sys.exit(__doc__)
    casos[sys.argv[1]]()
