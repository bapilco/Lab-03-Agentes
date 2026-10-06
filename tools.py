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

def execute_query(
    query: str,
    db_path: str = "data/database.sqlite",
) -> dict:
    """
    Ejecuta una consulta SQL de solo lectura.

    Regla 2:
    - SQLite se abre en modo read-only.
    - Solo se permite una sentencia.
    - Solo SELECT o WITH ... SELECT.

    Regla 3:
    - El límite máximo de filas lo impone el código.
    - El modelo no recibe row_limit como argumento.
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "La consulta SQL debe ser un texto no vacío."}

    sql = query.strip()

    # No permitir múltiples sentencias.
    if ";" in sql.rstrip(";"):
        return {"error": "Solo se permite una sentencia SQL."}

    normalized = sql.lower().strip()

    # Solo SELECT o WITH.
    if not (
        normalized.startswith("select ")
        or normalized == "select"
        or normalized.startswith("with ")
    ):
        return {"error": "Solo se permiten consultas SELECT o WITH ... SELECT."}

    path = Path(db_path)
    if not path.exists():
        return {"error": f"No existe la base de datos: {db_path}"}

    # El límite lo impone el código, no el modelo.
    limited_query = f"""
        SELECT *
        FROM ({sql.rstrip(";")})
        LIMIT {DEFAULT_ROW_LIMIT}
    """

    try:
        # SQLite en modo solo lectura.
        with sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            uri=True,
        ) as conn:
            frame = pd.read_sql_query(limited_query, conn)

        return {
            "columns": list(frame.columns),
            "rows": frame.to_dict(orient="records"),
            "row_count": len(frame),
        }

    except Exception as exc:
        return {"error": f"Error ejecutando la consulta: {exc}"}

def compute_statistics(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> dict:
    """
    Calcula estadísticas descriptivas sobre los datos recibidos.

    Regla 4:
    - Recibe los datos directamente desde execute_query.
    - No depende de un archivo CSV externo.

    Regla 5:
    - Valida los argumentos antes de procesarlos.
    - Los errores se devuelven como {"error": "..."}.
    """
    if not isinstance(rows, list):
        return {"error": "rows debe ser una lista de registros."}

    if not rows:
        return {"error": "No hay datos para calcular estadísticas."}

    try:
        frame = pd.DataFrame(rows)

        if frame.empty:
            return {"error": "No hay datos para calcular estadísticas."}

        numeric = frame.select_dtypes(include="number")

        if columns:
            if not isinstance(columns, list):
                return {"error": "columns debe ser una lista de nombres de columnas."}

            missing = [column for column in columns if column not in numeric.columns]

            if missing:
                return {
                    "error": f"Columnas numéricas no encontradas: {missing}"
                }

            numeric = numeric[columns]

        if numeric.empty:
            return {
                "error": "No se encontraron columnas numéricas para calcular estadísticas."
            }

        return {
            "statistics": numeric.describe().to_dict()
        }

    except Exception as exc:
        return {
            "error": f"Error calculando estadísticas: {exc}"
        }

def generate_chart(
    rows: list[dict[str, Any]],
    x: str,
    y: str,
    chart_type: str = "line",
) -> dict:
    """
    Genera un gráfico a partir de los datos recibidos.

    Regla 3:
    - La ruta de salida NO es un argumento controlable por el modelo.
    - El gráfico siempre se guarda dentro de traces/.

    Regla 4:
    - Recibe los datos directamente desde execute_query().

    Regla 5:
    - Valida los argumentos antes de generar el gráfico.
    - Los errores se devuelven como {"error": "..."}.
    """

    if not isinstance(rows, list):
        return {"error": "rows debe ser una lista de registros."}

    if not rows:
        return {"error": "No hay datos para generar el gráfico."}

    if not isinstance(x, str) or not isinstance(y, str):
        return {
            "error": "x e y deben ser nombres de columnas."
        }

    if chart_type not in {"line", "bar", "scatter"}:
        return {
            "error": (
                "Tipo de gráfico inválido. "
                "Use: line, bar o scatter."
            )
        }

    try:
        frame = pd.DataFrame(rows)

        if frame.empty:
            return {"error": "No hay datos para generar el gráfico."}

        if x not in frame.columns or y not in frame.columns:
            return {
                "error": (
                    f"Columnas inválidas. "
                    f"Disponibles: {list(frame.columns)}"
                )
            }

        # La ruta es controlada por el código.
        output_dir = Path("traces")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "chart.png"

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
        plt.savefig(output_path)
        plt.close()

        return {
            "chart_path": str(output_path),
            "chart_type": chart_type,
            "x": x,
            "y": y,
        }

    except Exception as exc:
        return {
            "error": f"Error generando gráfico: {exc}"
        }

TOOL_REGISTRY = {
    "execute_query": execute_query,
    "compute_statistics": compute_statistics,
    "generate_chart": generate_chart,
}
