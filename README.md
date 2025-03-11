# Research Assistant

An advanced AI-powered research assistant that uses multiple AI agents to conduct comprehensive research on any topic.

## Features

- Automated research using AI agents
- Comprehensive analysis and visualization
- Interactive web interface
- Real-time report generation
- Support for mermaid diagrams and LaTeX equations

## Requirements

- Python 3.8+
- Streamlit
- CrewAI
- OpenAI API key
- Serper API key

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:

```
SERPER_API_KEY=your_serper_api_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4-turbo  # or your preferred model
```

## Local Development

```bash
streamlit run app.py
```

## Deployment

### Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy your app by selecting the repository
5. Add your environment variables in the Streamlit Cloud dashboard:
   - SERPER_API_KEY
   - OPENAI_API_KEY
   - OPENAI_MODEL

## Usage

1. Enter your research topic
2. Configure API keys and model settings
3. Click "Start Research"
4. View the generated report with visualizations
