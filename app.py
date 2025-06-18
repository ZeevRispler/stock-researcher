import gradio as gr
from workflow import Workflow

workflow = Workflow()


def run_research(query: str):
    if not query:
        return "Please enter a query to begin."

    try:
        final_state = workflow.run(query)

        if final_state.get("error_messages"):
            return f"An error occurred: {final_state['error_messages'][0]}"

        output = ""
        if final_state.get("executive_summary"):
            output += f"## 📝 Executive Summary\n\n{final_state['executive_summary']}\n\n"

        if final_state.get("comparison_dashboard"):
            output += f"## 📊 Comparison Dashboard\n\n```\n{final_state['comparison_dashboard']}\n```"

        return output

    except Exception as e:
        return f"A critical error occurred: {str(e)}"


iface = gr.Interface(
    fn=run_research,
    inputs=gr.Textbox(
        lines=3,
        placeholder="e.g., 'Analyze Nvidia stock' or 'Compare TSLA vs RIVN'",
        label="Query",
    ),
    outputs=gr.Markdown(label="Research Report", elem_id="markdown-output"),
    title="🤖 Multi-Agent Stock Research",
    description="Enter a query to research a single stock or compare two. The system uses multiple agents to parse, gather, analyze, and report.",
    examples=[
        ["Analyze GOOGL"],
        ["Compare Apple and Microsoft"],
        ["NVDA vs AMD stock analysis"],
    ],
    allow_flagging="never",
    css="#markdown-output { white-space: pre-wrap; }",
)

if __name__ == "__main__":
    iface.launch()