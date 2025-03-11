import sys
import streamlit as st
import os
from dotenv import load_dotenv

# Fix for SQLite version issue
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import json
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
import tempfile
import time
import re
from streamlit_mermaid import st_mermaid

load_dotenv()

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #0D47A1;
    }
    .info-text {
        font-size: 1rem;
        color: #FFFFFF;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .success {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    .progress {
        background-color: #FFF8E1;
        color: #F57F17;
    }
</style>
""", unsafe_allow_html=True)

class VisualizationTools:
    @staticmethod
    def extract_mermaid_diagrams(markdown_content):
        """Extract Mermaid diagrams from markdown content"""
        pattern = r"```mermaid\n(.*?)```"
        return re.findall(pattern, markdown_content, re.DOTALL)

    @staticmethod
    def extract_latex_equations(markdown_content):
        """Extract LaTeX equations from markdown content"""
        inline_pattern = r"\$([^\$]+)\$"
        block_pattern = r"\$\$(.*?)\$\$"
        inline_matches = re.findall(inline_pattern, markdown_content)
        block_matches = re.findall(block_pattern, markdown_content, re.DOTALL)
        return {"inline": inline_matches, "block": block_matches}

    @staticmethod
    def render_mermaid_diagram(diagram_code, key=None):
        """Render a Mermaid diagram using streamlit-mermaid"""
        diagram_code = diagram_code.strip()
        if diagram_code.startswith("```mermaid"):
            diagram_code = diagram_code[10:].strip()
        if diagram_code.endswith("```"):
            diagram_code = diagram_code[:-3].strip()
        st_mermaid(diagram_code, key=key)

    @staticmethod
    def render_latex(equation, block=False, key=None):
        """Render a LaTeX equation"""
        if block:
            st.latex(equation)
        else:
            st.markdown(f"${equation}$")

    @staticmethod
    def create_relationship_graph(nodes, edges):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        plt.figure(figsize=(12, 8))
        nx.draw(G, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

class ResearchTools:
    def __init__(self):
        self.search_tool = SerperDevTool()

    def fetch_citations(self, query):
        try:
            from scholarly import scholarly
            search_query = scholarly.search_pubs(query)
            citations = []
            try:
                for i in range(5): 
                    pub = next(search_query)
                    citations.append({
                        'title': pub.bib.get('title', ''),
                        'author': pub.bib.get('author', ''),
                        'year': pub.bib.get('year', '')
                    })
            except StopIteration:
                pass
            return citations
        except ImportError:
            st.warning("The 'scholarly' package is not installed. Citations cannot be fetched.")
            return []

def get_api_keys():
    """Get API keys from environment or user input"""
    serper_key = os.getenv("SERPER_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    try:
        if not serper_key:
            serper_key = st.secrets.get("SERPER_API_KEY", "")
        if not openai_key:
            openai_key = st.secrets.get("OPENAI_API_KEY", "")
        if openai_model == "gpt-4o-mini":
            openai_model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    except:
        pass
    
    return serper_key, openai_key, openai_model

def setup_agents():
    """Setup and return the agents required for the research"""
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
    
    return researcher, visualizer, writer

def create_research_tasks(topic, researcher, visualizer, writer):
    """Create and return the tasks for the research crew"""
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

def generate_report(topic, status_placeholder, serper_key, openai_key, openai_model):
    """Generate a research report on the given topic"""
    os.environ["SERPER_API_KEY"] = serper_key
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["OPENAI_MODEL"] = openai_model
    
    researcher, visualizer, writer = setup_agents()
    
    tasks = create_research_tasks(topic, researcher, visualizer, writer)
    
    crew = Crew(
        agents=[researcher, visualizer, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True 
    )
    
    status_placeholder.markdown('<p class="status progress">Starting the research process...</p>', unsafe_allow_html=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, f"research_report_{datetime.now().strftime('%Y%m%d')}.md")
        tasks[-1].output_file = output_file
        
        status_placeholder.markdown('<p class="status progress">Research in progress...</p>', unsafe_allow_html=True)
        result = crew.kickoff(inputs={'topic': topic})
        
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                report_content = f.read()
        else:
            report_content = result
    
    status_placeholder.markdown('<p class="status success">Research completed successfully!</p>', unsafe_allow_html=True)
    
    return report_content

def render_report_with_visualizations(report_content):
    """Render the report content with compiled visualizations"""
    sections = []
    current_pos = 0
    
    mermaid_matches = list(re.finditer(r"```mermaid.*?```", report_content, re.DOTALL))
    latex_matches = list(re.finditer(r"\$\$.*?\$\$", report_content, re.DOTALL))
    
    all_matches = sorted(mermaid_matches + latex_matches, key=lambda x: x.start())
    
    for match in all_matches:
        if match.start() > current_pos:
            sections.append(("text", report_content[current_pos:match.start()]))
        

        content = match.group()
        if content.startswith("```mermaid"):
            sections.append(("mermaid", content))
        else:
            sections.append(("latex", content))
        
        current_pos = match.end()
    
    if current_pos < len(report_content):
        sections.append(("text", report_content[current_pos:]))
    
    for idx, (section_type, content) in enumerate(sections):
        display_id = f"section_{idx}"
        
        section_container = st.container()
        
        with section_container:
            if section_type == "mermaid":
                VisualizationTools.render_mermaid_diagram(content, key=f"report_mermaid_{idx}")
            elif section_type == "latex":
                equation = content[2:-2].strip()
                VisualizationTools.render_latex(equation, block=True)
            else:
                content = re.sub(r"\$([^\$]+)\$", r"\\(\1\\)", content)
                st.markdown(content, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🔍 Advanced Research Assistant</h1>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ About this app", expanded=False):
        st.markdown("""
        <div class="info-text">
        <p>This application uses AI agents to conduct comprehensive research on any topic you provide. 
        The system employs three specialized AI agents:</p>
        <ul>
            <li><strong>Research Analyst:</strong> Gathers and analyzes information</li>
            <li><strong>Data Visualization Specialist:</strong> Creates diagrams and visual representations</li>
            <li><strong>Technical Writer:</strong> Compiles findings into a well-structured report</li>
        </ul>
        <p>Simply enter your research topic, configure the optional settings, and click "Start Research" to begin.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<h2 class="sub-header">API Configuration</h2>', unsafe_allow_html=True)
        
        serper_key, openai_key, openai_model = get_api_keys()
        
        serper_key_input = st.text_input("Serper API Key", value=serper_key, type="password", 
                                    help="Required for web search capability")
        
        openai_key_input = st.text_input("OpenAI API Key", value=openai_key, type="password", 
                                    help="Required for the AI agents")
        
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        selected_model = st.selectbox("OpenAI Model", options=model_options, 
                                    index=model_options.index(openai_model) if openai_model in model_options else 0)
        
        st.divider()
        
        st.markdown('<h2 class="sub-header">Research Settings</h2>', unsafe_allow_html=True)
        research_depth = st.slider("Research Depth", min_value=1, max_value=5, value=3, 
                                  help="Higher values produce more detailed reports but take longer")
        
        include_visualizations = st.toggle("Include Visualizations", value=True)
        include_citations = st.toggle("Include Citations", value=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Research Topic</h2>', unsafe_allow_html=True)
        topic = st.text_area("Enter your research topic or question", 
                            placeholder="E.g., Advancements in quantum computing and its potential applications",
                            height=100)
        
        st.markdown('<h2 class="sub-header">Additional Context (Optional)</h2>', unsafe_allow_html=True)
        additional_context = st.text_area("Provide any additional context or specific aspects to focus on",
                                         placeholder="E.g., Focus on business applications rather than technical details",
                                         height=100)
    
    with col2:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Research Overview</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">Your research will include:</p>
        <ul>
            <li>Comprehensive analysis</li>
            <li>Latest developments</li>
            <li>Key players & technologies</li>
            <li>Market analysis</li>
            <li>Future outlook</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    final_topic = topic
    if additional_context.strip():
        final_topic = f"{topic}. Additional context: {additional_context}"
    
    status_placeholder = st.empty()
    
    serper_key = serper_key_input
    openai_key = openai_key_input
    openai_model = selected_model
    
    if st.button("🚀 Start Research", disabled=not topic or not serper_key or not openai_key):
        report_container = st.container()
        
        with st.spinner("Research in progress... This may take several minutes."):
            try:
                report_content = generate_report(final_topic, status_placeholder, 
                                                serper_key, openai_key, openai_model)
                
                with report_container:
                    st.markdown('<h2 class="sub-header">Research Report</h2>', unsafe_allow_html=True)
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Rendered Report", "Visualizations", "Equations", "Markdown Source"])
                    
                    with tab1:
                        render_report_with_visualizations(report_content)
                    
                    with tab2:
                        st.markdown("### Mermaid Diagrams")
                        mermaid_diagrams = VisualizationTools.extract_mermaid_diagrams(report_content)
                        for i, diagram in enumerate(mermaid_diagrams, 1):
                            st.markdown(f"#### Diagram {i}")
                            VisualizationTools.render_mermaid_diagram(diagram, key=f"mermaid_diagram_{i}")
                            st.markdown("---")
                    
                    with tab3:
                        st.markdown("### LaTeX Equations")
                        equations = VisualizationTools.extract_latex_equations(report_content)
                        if equations["block"]:
                            st.markdown("#### Block Equations")
                            for i, eq in enumerate(equations["block"], 1):
                                st.markdown(f"Equation {i}:")
                                VisualizationTools.render_latex(eq, block=True)
                                st.markdown("---")
                        if equations["inline"]:
                            st.markdown("#### Inline Equations")
                            for i, eq in enumerate(equations["inline"], 1):
                                st.markdown(f"Equation {i}: ")
                                VisualizationTools.render_latex(eq)
                                st.markdown("---")
                    
                    with tab4:
                        st.text_area("Markdown Source", report_content, height=500)
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report_content,
                        file_name=f"research_report_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
            
            except Exception as e:
                st.error(f"An error occurred during the research process: {str(e)}")
                status_placeholder.markdown(f'<p class="status error">Error: {str(e)}</p>', unsafe_allow_html=True)
    
    with st.expander("📚 Tips for effective research topics", expanded=False):
        st.markdown("""
        <div class="info-text">
        <p>For best results with your research topics:</p>
        <ul>
            <li><strong>Be specific:</strong> "Impact of AI on healthcare diagnostics" is better than "AI in healthcare"</li>
            <li><strong>Add time frames:</strong> "Renewable energy trends for 2025-2030" helps focus the research</li>
            <li><strong>Include multiple aspects:</strong> "Blockchain applications in supply chain: benefits, challenges, and case studies"</li>
            <li><strong>Ask for comparisons:</strong> "Compare quantum computing approaches: gate-based vs. quantum annealing"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🔧 Installation Requirements", expanded=False):
        st.markdown("""
        <div class="info-text">
        <p>To run this application, you need to install the following dependencies:</p>
        <pre>
pip install streamlit python-dotenv crewai crewai-tools langchain-openai networkx matplotlib pandas streamlit-mermaid
        </pre>
        
        <p>Optional dependencies for additional features:</p>
        <pre>
pip install scholarly mdutils
        </pre>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🛠️ Troubleshooting", expanded=False):
        st.markdown("""
        <div class="info-text">
        <p><strong>Common Issues:</strong></p>
        <ul>
            <li><strong>API Key Errors:</strong> Make sure your Serper and OpenAI API keys are valid</li>
            <li><strong>ImportError:</strong> Run the installation commands to install all required packages</li>
            <li><strong>Research Timeout:</strong> Some complex topics may take longer to research. Consider simplifying your query</li>
            <li><strong>Rate Limiting:</strong> If you encounter rate limit errors, wait a few minutes before trying again</li>
        </ul>
        
        <p><strong>CrewAI Version:</strong> This app is designed to work with CrewAI version 0.22.0 or higher.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p style="text-align:center">Advanced Research Assistant</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()