import json

from myagent_langgraph import MyAnalystAgentLangGraph


class FakeRepetitionAgent(MyAnalystAgentLangGraph):
    """
    Fuerza siempre la misma llamada a herramienta para probar
    el detector de repetición sin consumir llamadas reales al LLM.
    """

    def _call_llm(self, messages):
        assistant_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_repetition",
                    "type": "function",
                    "function": {
                        "name": "execute_query",
                        "arguments": json.dumps(
                            {
                                "query": "SELECT 1"
                            }
                        ),
                    },
                }
            ],
        }

        model_output = {
            "thought": "Voy a ejecutar nuevamente la misma consulta.",
            "action": {
                "name": "execute_query",
                "args": {
                    "query": "SELECT 1"
                },
            },
        }

        usage_data = {
            "tokens_entrada": 10,
            "tokens_salida": 5,
            "latencia_segundos": 0.01,
            "error": None,
        }

        return model_output, assistant_message, usage_data


if __name__ == "__main__":
    agent = FakeRepetitionAgent(
        max_steps=8,
        token_budget=10000,
        repeat_limit=3,
    )

    result = agent.run("Prueba del detector de repetición")

    print(result["status"])
    print(result["answer"])
    print("Pasos:", len(result["trace"]))
    print(result["usage"])

    print("\nTRACE:")
    print(
        json.dumps(
            result["trace"],
            indent=2,
            ensure_ascii=False,
        )
    )