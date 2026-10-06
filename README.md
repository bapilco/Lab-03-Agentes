# Lab 03 — Agentes LLM con Tool Use

> **Revisión del 2026-09-30.** El laboratorio trae ahora datos (`crear_base.py`), un
> evaluador que escribe el CSV crudo (`evaluar.py`), un golden set de ejemplo
> (`preguntas_plantilla.json`) y un modelo de guion para la Parte 0 del Taller 3
> (`guion.py`). **`agent.py` y `tools.py` no cambiaron**: son el andamiaje cuyas fallas la
> Parte 0 reproduce y la Parte 1 corrige. Ver `curso/talleres/taller-03-agentes.md`.

## Setup

```bash
pip install -r requirements.txt     # versiones fijadas; langgraph==1.2.12, la del curso
python crear_base.py                # data/database.sqlite y data/ventas_mensuales.csv
cp .env.example .env                # y elige una ruta de modelo (abajo)
```

## Qué hay

```
Lab-03-Agentes/
├── agent.py                 # el agente ReAct del andamiaje: bucle, JSON por prompt, traza
├── tools.py                 # tres herramientas: execute_query, compute_statistics, generate_chart
├── crear_base.py            # la base de ejemplo: 3 tablas, 6 000 ventas, semilla fija
├── evaluar.py               # corre un agente sobre un golden set → resultados.csv
├── preguntas_plantilla.json # 8 preguntas con su consulta de verificación
├── guion.py                 # un «modelo» fijo, sin LLM, para la Parte 0.b y 0.c
├── solver-v2/               # kit del Taller 03 v2 (opcional): enunciados, Parte 0, golden set, evaluador
├── requirements.txt
├── data/                    # lo escribe crear_base.py (no se versiona)
└── traces/                  # una traza JSON por corrida (no se versiona)
```

## Con qué modelo corre

`agent.py` construye `OpenAI()`, y el SDK lee `OPENAI_BASE_URL` del entorno: las tres rutas
corren **sin tocar el código**.

| Ruta | `.env` | Costo |
|---|---|---|
| Clave del curso | `OPENAI_API_KEY=…` y `AGENT_MODEL=` el `model` de la fila `propietario_economico` de `fuentes/modelos/modelos-2026-1.json` | el de la fila, con su fecha |
| H200 de la USFQ (GlobalProtect) | `OPENAI_BASE_URL=http://172.28.230.10:12555/v1`, `OPENAI_API_KEY=local` y `AGENT_MODEL=` el id que devuelve `/v1/models` | cero |
| Ollama en tu máquina | `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama` y el modelo descargado | cero |

El id del modelo **no se escribe a mano**: sale de la tabla semestral o del endpoint. La
H200 cambió el modelo que sirve al menos una vez este semestre, y un id escrito devuelve 404.

## Lo que el andamiaje hace mal a propósito

No es un descuido que esté en el código que se reparte: es lo que la Parte 0 del taller te
pide reproducir y la Parte 1 corregir. Están catalogados en `curso/planificacion/plan.md`
(`0.af`, `0.ag`, `0.ak`, `0.al`, `0.am`, `0.an`):

- el prompt no enumera las herramientas, así que el modelo no las usa o se inventa otras;
- el guard de SQL compara subcadenas con espacios, y ninguna capa captura una excepción;
- la traza se escribe solo si la corrida termina, y no guarda modelo, tokens ni tiempos;
- `row_limit` y `output_path` son parámetros públicos: el límite lo fija quien llama;
- `execute_query` lee SQLite y las otras dos leen CSV, y nada convierte una cosa en la otra.

## Evaluar

```bash
python evaluar.py                                   # el andamiaje, sobre la plantilla
python evaluar.py --golden golden_set.json --agente mi_agente:MiAgente
```

Una fila por pregunta en `resultados.csv`: estado, pasos, llamadas y errores de herramienta,
excepción, tokens, segundos, valor esperado, acierto, abstención y si la base sigue intacta.
El contrato con el agente es el mismo que usa el Lab 04: `.run(pregunta)` devuelve un `dict`
con `answer`, `trace` y `status`.

## Recursos

- LangGraph v1, *Graph API*: https://docs.langchain.com/oss/python/langgraph/graph-api (la
  documentación anterior a v1 redirige; ver `fuentes/web/fuentes-web.md`)
- Anthropic tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- OpenAI function calling: https://platform.openai.com/docs/guides/function-calling
- MCP, revisión 2026-07-28: https://modelcontextprotocol.io/specification/2026-07-28
