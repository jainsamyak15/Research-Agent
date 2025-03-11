import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import json
from datetime import datetime
from mdutils.mdutils import MdUtils
import networkx as nx
import matplotlib.pyplot as plt
from scholarly import scholarly
import pandas as pd

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

class VisualizationTools:
    @staticmethod
    def create_mermaid_diagram(data):
        return f"""```mermaid
graph TD
    {data}
```"""

    @staticmethod
    def create_relationship_graph(nodes, edges):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        plt.figure(figsize=(12, 8))
        nx.draw(G, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        plt.savefig('relationship_graph.png')
        plt.close()

class ResearchTools:
    def __init__(self):
        self.search_tool = SerperDevTool()

    def fetch_citations(self, query):
        search_query = scholarly.search_pubs(query)
        citations = []
        try:
            for i in range(5):  # Get first 5 results
                pub = next(search_query)
                citations.append({
                    'title': pub.bib.get('title', ''),
                    'author': pub.bib.get('author', ''),
                    'year': pub.bib.get('year', '')
                })
        except StopIteration:
            pass
        return citations

# Enhanced Researcher Agent
researcher = Agent(
    role='Senior Research Analyst',
    goal='Conduct comprehensive research and create detailed analysis with visualizations',
    verbose=True,
    memory=True,
    backstory="""You are an elite research analyst with expertise in creating 
    comprehensive reports with data visualization. You excel at identifying patterns,
    creating relationships between concepts, and presenting information in an 
    engaging and visually appealing manner.""",
    tools=[SerperDevTool()],
    allow_delegation=True
)

# Data Visualization Specialist Agent
visualizer = Agent(
    role='Data Visualization Specialist',
    goal='Create compelling visualizations and diagrams from research data',
    verbose=True,
    memory=True,
    backstory="""You specialize in transforming complex data into clear, 
    visually appealing diagrams and charts. You have expertise in creating 
    mermaid diagrams, relationship graphs, and other visual representations.""",
    tools=[SerperDevTool()],
    allow_delegation=False
)

# Technical Writer Agent
writer = Agent(
    role='Technical Writer',
    goal='Create comprehensive and well-structured technical documentation',
    verbose=True,
    memory=True,
    backstory="""You are an experienced technical writer who excels at 
    creating clear, engaging, and well-organized documentation. You know 
    how to present complex information in an accessible format.""",
    tools=[SerperDevTool()],
    allow_delegation=False
)

def create_research_tasks(topic):
    research_task = Task(
        description=f"""Conduct comprehensive research on {topic}. Include:
        1. Latest developments and trends
        2. Key players and technologies
        3. Market analysis and future predictions
        4. Potential challenges and solutions
        5. Related research papers and citations
        
        Format the findings in a structured JSON format for visualization.""",
        agent=researcher,
        expected_output="Detailed research findings in JSON format"
    )

    visualization_task = Task(
        description=f"""Create visual representations for the research on {topic}:
        1. Generate a mermaid diagram showing the relationship between key concepts
        2. Create a relationship graph of key players and technologies
        3. Design a timeline of developments
        4. Visualize market trends and predictions""",
        agent=visualizer,
        expected_output="A collection of visual elements in markdown format"
    )

    writing_task = Task(
        description=f"""Create a comprehensive report on {topic} including:
        1. Executive Summary
        2. Detailed Analysis
        3. Visual Elements
        4. Citations and References
        5. Future Outlook
        
        Format the report in markdown with clear sections and styling.""",
        agent=writer,
        expected_output="A complete markdown report with all elements integrated",
        output_file=f"research_report_{datetime.now().strftime('%Y%m%d')}.md"
    )

    return [research_task, visualization_task, writing_task]

def generate_report(topic):
    tasks = create_research_tasks(topic)
    crew = Crew(
        agents=[researcher, visualizer, writer],
        tasks=tasks,
        process=Process.sequential
    )
    
    result = crew.kickoff(inputs={'topic': topic})
    return result

if __name__ == "__main__":
    print("🔍 Advanced Research Agent System")
    print("================================")
    topic = input("Enter research topic: ")
    print("\nInitiating comprehensive research process...")
    result = generate_report(topic)
    print("\n✨ Research complete! Check the generated report file.")