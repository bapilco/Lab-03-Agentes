from myagent_langgraph import MyAnalystAgentLangGraph


agent = MyAnalystAgentLangGraph()

mermaid = agent.graph.get_graph().draw_mermaid()

with open("diagrama_langgraph.md", "w", encoding="utf-8") as f:
    f.write("```mermaid\n")
    f.write(mermaid)
    f.write("\n```\n")

print("Diagrama generado en diagrama_langgraph.md")
print(mermaid)