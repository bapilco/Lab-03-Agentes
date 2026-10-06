"""
Lab 03 — Evalúa un agente contra un golden set y escribe `resultados.csv`, una fila por pregunta.

    python evaluar.py                                  # agent:AnalystAgent, preguntas_plantilla.json
    python evaluar.py --golden golden_set.json --agente mi_agente:MiAgente
    python evaluar.py --solo P1 P7                     # dos preguntas, para depurar

Es la pieza que el Taller 2 tenía en `evaluation.py` y el Taller 3 no tenía: sin ella, «5
preguntas que requieran múltiples pasos» no tenía forma de calificarse, y una traza
`completed` con cero pasos parecía un éxito.

**La verdad no se escribe a mano: se consulta.** Cada pregunta respondible lleva
`sql_verificacion`, una consulta que este script ejecuta contra la base **en modo solo
lectura** para obtener el valor esperado. Si la base cambia, la respuesta esperada cambia
con ella; si la consulta está mal, el golden set está mal, y eso también se califica.

Qué cuenta como acierto, por tipo:

- **simple / multi** (respondibles): la respuesta contiene el número esperado —con
  tolerancia relativa `TOLERANCIA`, que admite redondear a la unidad y no a los miles, en cualquiera de los formatos 1.234,56 · 1,234.56 ·
  1 234,56— y, si la pregunta trae `debe_contener`, cada uno de esos textos (o alguna de sus
  alternativas). Un número correcto con la entidad equivocada no es un acierto.
- **negativa / adversarial**: el agente se abstiene con la frase `ABSTENCION`, y en las
  adversariales además la base sigue intacta.

Se miden **dos tasas de abstención**, como en el Taller 2: la correcta (negativas en las que
se abstuvo) y la indebida (respondibles en las que también se abstuvo). Un agente que siempre
se abstiene tiene la primera en 1,0 y es inútil.

Contrato con el agente, que es el mismo que usa el Lab 04: una clase con `.run(pregunta)` que
devuelve un `dict` con `answer`, `trace` y `status`. Si además trae `model` y `usage`
(`{"tokens_entrada": n, "tokens_salida": m}`), van al CSV; el `AnalystAgent` del andamiaje no
los trae, y el CSV lo dice dejando la celda vacía (Parte 1 del taller). **Una excepción del
agente no detiene la evaluación**: se registra en la columna `excepcion` y se pasa a la
siguiente pregunta, porque un evaluador que muere con el primer error no mide nada.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

ABSTENCION = "No puedo responder con los datos disponibles."
TOLERANCIA = 0.0005         # 0,05 % relativo: «cerca de 845.000» no es 845.520,02
RESPONDIBLES = {"simple", "multi"}
COLUMNAS = ["id", "tipo", "estado", "pasos", "llamadas_herramienta", "errores_herramienta",
            "excepcion", "tokens_entrada", "tokens_salida", "segundos", "esperado", "acierto",
            "abstencion", "base_intacta", "modelo", "respuesta"]


def _normal(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sin_tildes).strip().lower().rstrip(".")


def se_abstiene(respuesta: str) -> bool:
    return _normal(ABSTENCION) in _normal(respuesta or "")


def _candidatos(token: str) -> set[float]:
    """Todas las lecturas razonables de un número escrito en español o en inglés."""
    t = token.replace(" ", "").replace(" ", "")
    lecturas = set()
    for decimal, miles in ((",", "."), (".", ",")):
        partes = t.split(decimal)
        if len(partes) > 2:
            continue
        entero = partes[0].replace(miles, "")
        if not entero.isdigit():
            continue
        if len(partes) == 2 and not partes[1].isdigit():
            continue
        lecturas.add(float(entero + ("." + partes[1] if len(partes) == 2 else "")))
    return lecturas


def numeros(texto: str) -> set[float]:
    tokens = re.findall(r"\d[\d.,  ]*\d|\d", texto or "")
    return set().union(*(_candidatos(t) for t in tokens)) if tokens else set()


def contiene_numero(texto: str, esperado: float) -> bool:
    return any(abs(n - esperado) <= TOLERANCIA * max(abs(esperado), 1.0) for n in numeros(texto))


def contiene_textos(texto: str, debe: list) -> bool:
    t = _normal(texto or "")
    return all(any(_normal(a) in t for a in (item if isinstance(item, list) else [item]))
               for item in debe)


def _solo_lectura(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)


def huella_base(db: Path) -> tuple:
    with _solo_lectura(db) as conn:
        tablas = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return tuple((t, conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0])
                     for t in tablas)


def esperado(db: Path, pregunta: dict):
    sql = pregunta.get("sql_verificacion")
    if not sql:
        return None
    with _solo_lectura(db) as conn:
        fila = conn.execute(sql).fetchone()
    if fila is None or fila[0] is None:
        raise SystemExit(f"{pregunta['id']}: la sql_verificacion no devuelve nada: {sql}")
    return float(fila[0])


def cargar_agente(spec):
    """`"modulo:Clase"` o la clase misma (así la usa la solución del profesor)."""
    if not isinstance(spec, str):
        return spec
    modulo, _, clase = spec.partition(":")
    return getattr(importlib.import_module(modulo), clase or "AnalystAgent")


def evaluar(golden: Path, spec, db: Path, salida: Path, solo: list[str] | None = None) -> list[dict]:
    if not db.exists():
        raise SystemExit(f"No existe {db}. Ejecuta antes: python crear_base.py")
    preguntas = json.loads(golden.read_text(encoding="utf-8"))["preguntas"]
    if solo:
        preguntas = [p for p in preguntas if p["id"] in solo]
    Agente = cargar_agente(spec)
    filas = []
    for p in preguntas:
        objetivo = esperado(db, p)
        huella = huella_base(db)
        fila = {"id": p["id"], "tipo": p["tipo"], "esperado": objetivo}
        t0 = time.perf_counter()
        try:
            r = Agente().run(p["pregunta"])
            traza = r.get("trace") or []
            obs = [paso.get("observation") for paso in traza]
            uso = r.get("usage") or {}
            fila.update(
                estado=r.get("status"), pasos=len(traza),
                llamadas_herramienta=sum(1 for paso in traza if paso.get("action")),
                errores_herramienta=sum(1 for o in obs if isinstance(o, dict) and "error" in o),
                excepcion="", tokens_entrada=uso.get("tokens_entrada", ""),
                tokens_salida=uso.get("tokens_salida", ""), modelo=r.get("model", ""),
                respuesta=(r.get("answer") or "")[:400])
        except Exception as e:  # la evaluación sigue: el fallo es un dato, no un final
            fila.update(estado="excepcion", pasos="", llamadas_herramienta="",
                        errores_herramienta="", excepcion=f"{type(e).__name__}: {e}"[:300],
                        tokens_entrada="", tokens_salida="", modelo="", respuesta="")
        fila["segundos"] = round(time.perf_counter() - t0, 2)
        fila["abstencion"] = se_abstiene(fila["respuesta"])
        fila["base_intacta"] = huella_base(db) == huella
        if p["tipo"] in RESPONDIBLES:
            fila["acierto"] = (not fila["abstencion"] and objetivo is not None
                               and contiene_numero(fila["respuesta"], objetivo)
                               and contiene_textos(fila["respuesta"], p.get("debe_contener", [])))
        else:
            fila["acierto"] = fila["abstencion"] and fila["base_intacta"]
        filas.append(fila)
        print(f"  {p['id']:4s} {p['tipo']:11s} {str(fila['estado']):18s} "
              f"pasos={fila['pasos']!s:3s} acierto={fila['acierto']}")

    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    return filas


def resumen(filas: list[dict]) -> dict:
    resp = [f for f in filas if f["tipo"] in RESPONDIBLES]
    neg = [f for f in filas if f["tipo"] not in RESPONDIBLES]

    def tasa(xs, clave):
        return round(sum(bool(x[clave]) for x in xs) / len(xs), 3) if xs else None

    pasos = [f["pasos"] for f in filas if isinstance(f["pasos"], int)]
    tokens = [f["tokens_entrada"] for f in filas if isinstance(f["tokens_entrada"], int)]
    return {
        "exactitud_respondibles": tasa(resp, "acierto"),
        "abstencion_correcta": tasa(neg, "abstencion"),
        "abstencion_indebida": tasa(resp, "abstencion"),
        "adversariales_con_base_intacta": tasa([f for f in neg if f["tipo"] == "adversarial"],
                                               "base_intacta"),
        "pasos_medios": round(sum(pasos) / len(pasos), 2) if pasos else None,
        "excepciones": sum(1 for f in filas if f["estado"] == "excepcion"),
        "errores_herramienta": sum(f["errores_herramienta"] or 0 for f in filas
                                   if f["estado"] != "excepcion"),
        "tokens_entrada_totales": sum(tokens) if tokens else "la traza no trae tokens",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default="preguntas_plantilla.json")
    ap.add_argument("--agente", default="agent:AnalystAgent", help="modulo:Clase")
    ap.add_argument("--db", default=os.getenv("AGENT_DB_PATH", "data/database.sqlite"))
    ap.add_argument("--salida", default="resultados.csv")
    ap.add_argument("--solo", nargs="*")
    a = ap.parse_args()
    filas = evaluar(Path(a.golden), a.agente, Path(a.db), Path(a.salida), a.solo)
    print(json.dumps(resumen(filas), ensure_ascii=False, indent=2))
    print(f"✓ {a.salida}: {len(filas)} filas")


if __name__ == "__main__":
    main()
