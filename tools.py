"""
Lab 03 — Tools baseline para un agente analitico.

Estas funciones son deliberadamente conservadoras: solo lectura, limites de filas y
salidas compactas para controlar costo y riesgo.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROW_LIMIT = 50


def execute_query(query: str, db_path: str = "data/database.sqlite", row_limit: int = DEFAULT_ROW_LIMIT) -> dict:
    normalized = query.strip().lower()
    if not normalized.startswith("select"):
        return {"error": "Solo se permiten consultas SELECT en este lab."}
    if any(token in normalized for token in [" insert ", " update ", " delete ", " drop ", " alter "]):
        return {"error": "Consulta rechazada por contener operacion de escritura o schema."}

    path = Path(db_path)
    if not path.exists():
        return {"error": f"No existe la base de datos: {db_path}"}

    limited_query = query
    if " limit " not in normalized:
        limited_query = f"{query.rstrip(';')} LIMIT {row_limit}"

    with sqlite3.connect(path) as conn:
        frame = pd.read_sql_query(limited_query, conn)
    return {
        "columns": list(frame.columns),
        "rows": frame.to_dict(orient="records"),
        "row_count": len(frame),
    }


def compute_statistics(csv_path: str, columns: list[str] | None = None) -> dict:
    path = Path(csv_path)
    if not path.exists():
        return {"error": f"No existe el archivo: {csv_path}"}
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include="number")
    if columns:
        missing = [column for column in columns if column not in numeric.columns]
        if missing:
            return {"error": f"Columnas numericas no encontradas: {missing}"}
        numeric = numeric[columns]
    return numeric.describe().to_dict()


def generate_chart(
    csv_path: str,
    x: str,
    y: str,
    chart_type: str = "line",
    output_path: str = "traces/chart.png",
) -> dict:
    path = Path(csv_path)
    if not path.exists():
        return {"error": f"No existe el archivo: {csv_path}"}

    frame = pd.read_csv(path)
    if x not in frame.columns or y not in frame.columns:
        return {"error": f"Columnas invalidas. Disponibles: {list(frame.columns)}"}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    if chart_type == "bar":
        plt.bar(frame[x], frame[y])
    elif chart_type == "scatter":
        plt.scatter(frame[x], frame[y])
    else:
        plt.plot(frame[x], frame[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return {"chart_path": str(output), "chart_type": chart_type, "x": x, "y": y}


TOOL_REGISTRY = {
    "execute_query": execute_query,
    "compute_statistics": compute_statistics,
    "generate_chart": generate_chart,
}
