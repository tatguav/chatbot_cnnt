from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import PDFSearchTool
import os
import sys
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class ChatbotCnnt():
    """ChatbotCnnt crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Modelo LLM seguro por defecto
    llm_model = os.getenv("MODEL", "HuggingFaceH4/zephyr-7b-alpha")

    # Validar existencia del token de Hugging Face
    if not os.getenv("HUGGINGFACE_ACCESS_TOKEN"):
        print("[ERROR] Falta la variable de entorno HUGGINGFACE_ACCESS_TOKEN. Por favor, agrégala a tu .env o entorno.")
        sys.exit(1)

    pdf_tool = PDFSearchTool(
        pdf=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'knowledge', 'Codigo-nacional-de-transito.pdf'
        ),
        config=dict(
            llm=dict(
                provider="huggingface",
                config=dict(
                    model=llm_model
                ),
            ),
        )
    )

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def ingestor(self) -> Agent:
        return Agent(
            config=self.agents_config['ingestor'], # type: ignore[index]
            tools=[self.pdf_tool],
            llm=dict(
                provider="huggingface",
                config=dict(
                    model=self.llm_model
                ),
            ),
            verbose=True
        )

    @agent
    def responder(self) -> Agent:
        return Agent(
            config=self.agents_config['responder'], # type: ignore[index]
            tools=[self.pdf_tool],
            llm=dict(
                provider="huggingface",
                config=dict(
                    model=self.llm_model
                ),
            ),
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def load_pdf(self) -> Task:
        return Task(
            config=self.tasks_config['load_pdf'], # type: ignore[index]
        )

    @task
    def answer_question(self) -> Task:
        return Task(
            config=self.tasks_config['answer_question'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ChatbotCnnt crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
