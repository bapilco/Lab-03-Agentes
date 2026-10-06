"""
Lab 03 — La base de datos de ejemplo del Taller 3.

    python crear_base.py            # escribe data/database.sqlite y data/ventas_mensuales.csv

Hasta el 2026-09-30 el laboratorio no traía datos (plan.md, discrepancia 0.am): el README
dibujaba un `data/` que no existía y el agente no tenía nada que consultar. Este script lo
genera, **siempre igual** —semilla fija—, para que la Parte 0 del taller y su solución den
los mismos números en cualquier máquina. No se versiona el `.sqlite`: lo produce este
archivo.

Tres tablas, el mismo dominio que el cuaderno del miércoles (licencia, soporte,
capacitación), y más de 5 000 ventas entre enero de 2025 y agosto de 2026:

    productos (producto_id, nombre, categoria, precio_lista)
    clientes  (cliente_id, nombre, region, segmento, fecha_alta)
    ventas    (venta_id, fecha, cliente_id, producto_id, cantidad, precio_unitario, descuento)

El importe de una venta es `cantidad * precio_unitario * (1 - descuento)`. Esa fórmula no
está en ninguna columna a propósito: es lo primero que el agente tiene que descubrir, y lo
que la pregunta de verificación de cada caso del golden set escribe explícitamente.

El CSV de totales mensuales existe porque `compute_statistics` y `generate_chart` leen
**CSV** y `execute_query` lee **SQLite**: las tres herramientas no comparten fuente, y
ninguna convierte el resultado de una consulta en un archivo. El CSV permite correr las dos
del CSV desde el primer día; el puente entre consulta y CSV es una decisión de la Parte 1.
"""

from __future__ import annotations

import csv
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SEMILLA = 6013
N_VENTAS = 6000
INICIO, FIN = date(2025, 1, 1), date(2026, 8, 31)

PRODUCTOS = [
    # (nombre, categoría, precio de lista en USD)
    ("Licencia Analítica Básica", "licencia", 480.0),
    ("Licencia Analítica Pro", "licencia", 1250.0),
    ("Licencia Analítica Enterprise", "licencia", 4200.0),
    ("Licencia Visualización", "licencia", 690.0),
    ("Licencia Integración API", "licencia", 980.0),
    ("Soporte Estándar (mes)", "soporte", 150.0),
    ("Soporte Prioritario (mes)", "soporte", 390.0),
    ("Soporte Dedicado (mes)", "soporte", 1100.0),
    ("Bolsa de horas de consultoría", "soporte", 85.0),
    ("Capacitación Fundamentos", "capacitación", 320.0),
    ("Capacitación Avanzada", "capacitación", 560.0),
    ("Taller in situ (día)", "capacitación", 1800.0),
    ("Certificación de analistas", "capacitación", 740.0),
]
REGIONES = ["Sierra", "Costa", "Amazonía", "Galápagos"]
PESO_REGION = [0.48, 0.38, 0.10, 0.04]
SEGMENTOS = ["pyme", "corporativo", "sector público", "educación"]
N_CLIENTES = 320


def _fecha_aleatoria(rng: random.Random) -> date:
    dias = (FIN - INICIO).days
    # Estacionalidad suave: más ventas en el último mes de cada trimestre.
    while True:
        d = INICIO + timedelta(days=rng.randrange(dias + 1))
        if d.month in (3, 6, 9, 12) or rng.random() < 0.78:
            return d


def crear(destino: Path = Path("data")) -> Path:
    rng = random.Random(SEMILLA)
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / "database.sqlite"
    if ruta.exists():
        ruta.unlink()

    with sqlite3.connect(ruta) as conn:
        conn.executescript("""
            CREATE TABLE productos (
                producto_id INTEGER PRIMARY KEY, nombre TEXT NOT NULL,
                categoria TEXT NOT NULL, precio_lista REAL NOT NULL);
            CREATE TABLE clientes (
                cliente_id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, region TEXT NOT NULL,
                segmento TEXT NOT NULL, fecha_alta TEXT NOT NULL);
            CREATE TABLE ventas (
                venta_id INTEGER PRIMARY KEY, fecha TEXT NOT NULL,
                cliente_id INTEGER NOT NULL REFERENCES clientes(cliente_id),
                producto_id INTEGER NOT NULL REFERENCES productos(producto_id),
                cantidad INTEGER NOT NULL, precio_unitario REAL NOT NULL,
                descuento REAL NOT NULL);
        """)
        conn.executemany("INSERT INTO productos VALUES (?, ?, ?, ?)",
                         [(i, n, c, p) for i, (n, c, p) in enumerate(PRODUCTOS, start=1)])

        clientes = []
        for i in range(1, N_CLIENTES + 1):
            region = rng.choices(REGIONES, PESO_REGION)[0]
            alta = date(2023, 1, 1) + timedelta(days=rng.randrange(900))
            clientes.append((i, f"Cliente {i:03d}", region, rng.choice(SEGMENTOS),
                             alta.isoformat()))
        conn.executemany("INSERT INTO clientes VALUES (?, ?, ?, ?, ?)", clientes)

        # La «Licencia Analítica Pro» crece de 2025 a 2026: hay una tendencia que encontrar.
        ventas = []
        for v in range(1, N_VENTAS + 1):
            d = _fecha_aleatoria(rng)
            pid = rng.randrange(1, len(PRODUCTOS) + 1)
            if d.year == 2026 and rng.random() < 0.10:
                pid = 2
            nombre, categoria, lista = PRODUCTOS[pid - 1]
            cantidad = rng.choice([1, 1, 1, 2, 2, 3, 5]) if categoria != "soporte" else \
                rng.choice([1, 3, 6, 12])
            precio = round(lista * rng.uniform(0.92, 1.05), 2)
            descuento = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15])
            ventas.append((v, d.isoformat(), rng.randrange(1, N_CLIENTES + 1), pid,
                           cantidad, precio, descuento))
        ventas.sort(key=lambda fila: fila[1])
        ventas = [(i, *fila[1:]) for i, fila in enumerate(ventas, start=1)]
        conn.executemany("INSERT INTO ventas VALUES (?, ?, ?, ?, ?, ?, ?)", ventas)

        mensual = conn.execute("""
            SELECT substr(fecha, 1, 7) AS mes,
                   ROUND(SUM(cantidad * precio_unitario * (1 - descuento)), 2) AS importe,
                   COUNT(*) AS n_ventas
            FROM ventas GROUP BY mes ORDER BY mes""").fetchall()

    with open(destino / "ventas_mensuales.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mes", "importe", "n_ventas"])
        w.writerows(mensual)
    return ruta


if __name__ == "__main__":
    ruta = crear()
    with sqlite3.connect(ruta) as conn:
        for tabla in ("productos", "clientes", "ventas"):
            n = conn.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
            print(f"  {tabla:10s} {n:6d} filas")
    print(f"✓ {ruta} y data/ventas_mensuales.csv (semilla {SEMILLA})")
