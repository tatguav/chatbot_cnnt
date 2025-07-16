# main.py
from crew import TransitoCrew
from crewai import Task

if __name__ == "__main__":
    # Pregunta de prueba (puedes cambiarla por cualquier otra)
    pregunta = "¿Qué es una acera?"

    # Crear instancia de la crew
    crew = TransitoCrew()

    # Ejecutar tarea con la pregunta
    resultado = crew.kickoff(tasks=[
        Task(
            name="answer_question",
            description="Responder pregunta sobre el Código Nacional de Tránsito",
            agent=crew.responder(),  # especifica el agente
            expected_output="Una respuesta legal clara y precisa.",
            input=pregunta
        )
    ])

    print("\n Respuesta del asistente:\n")
    print(resultado)
