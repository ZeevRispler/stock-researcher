# Multi-Agent Stock Research System

This project is a multi-agent stock research system built using LangGraph, Tavily, and OpenAI. It can understand natural language queries to research a single stock or compare two, performing data collection, news analysis, and risk assessment.

## Features

- **Natural Language Queries**: Ask questions like "Analyze NVDA" or "Compare Apple and Microsoft".
- **Multi-Agent Workflow**: Uses separate agents for parsing, data gathering, analysis, and synthesis.
- **Single & Comparison Analysis**: Generates an executive summary for single stocks and a comparison dashboard for two.
- **Data-Backed Analysis**: Fetches real-time information using the Tavily search API.
- **Built-in Validation**: Includes a validation agent to check the quality of the generated report.

## Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd stock-researcher
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    - Rename the `.env.example` file to `.env`.
    - Add your API keys for Tavily and OpenAI to the `.env` file.
    ```
    TAVILY_API_KEY="tvly-..."
    OPENAI_API_KEY="sk-..."
    ```

## Usage

You can run the system via a command-line interface or a Gradio web UI.

**1. Command-Line Interface (CLI)**

Execute `run_cli.py` with your query as an argument.

```bash
python run_cli.py "What is the risk profile for Tesla?"