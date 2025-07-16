# main.py
from crewai import Agent, Task, Crew
from crew import CustomRagTool, extraer_articulos
import os

# Ruta del PDF (relativa)
pdf_path = os.path.join("documents", "Codigo-nacional-de-transito.pdf")

# Extraer artículos
articulos = extraer_articulos(pdf_path)
herramienta = CustomRagTool(articulos)

# Crear agente
responder = Agent(
    role="Asistente legal de tránsito",
    goal="Responder preguntas legales usando el Código de Tránsito de Colombia",
    backstory="Especialista en normas de conducción, sanciones y regulación en Colombia.",
    tools=[herramienta]
)

# Crear tarea
pregunta_usuario = "¿Qué es una acera?"
tarea = Task(
    description=f"Responde esta pregunta: {pregunta_usuario}",
    expected_output="Respuesta legal clara y precisa para ciudadanos.",
    agent=responder
)

# Crear y ejecutar Crew
crew = Crew(
    agents=[responder],
    tasks=[tarea],
    verbose=True
)

resultado = crew.kickoff()
print("\n🔹 Respuesta del asistente:\n")
print(resultado)
