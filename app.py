import gradio as gr
from workflow import Workflow


# Create a single, reusable instance of the workflow
workflow = Workflow()


def run_research(query: str):
    """
    Wrapper function for the workflow to be used by the Gradio interface.
    It runs the complete workflow and formats the final output as Markdown.
    """
    if not query:
        return "Please enter a query to begin."

    try:
        final_state = workflow.run(query)

        if final_state.get("error_messages"):
            return f"An error occurred: {final_state['error_messages'][0]}"

        # Format the output for Gradio's Markdown component
        output = ""
        if final_state.get("executive_summary"):
            output += (
                "## 📝 Executive Summary\n\n"
                f"{final_state['executive_summary']}\n\n"
            )

        if final_state.get("comparison_dashboard"):
            output += (
                "## 📊 Comparison Dashboard\n\n"
                f"```\n{final_state['comparison_dashboard']}\n```\n\n"
            )

        output += "---\n\n### 🕵️ Agent Log\n\n"
        output += "\n".join([f"- {msg}" for msg in final_state.get("messages", [])])
        return output

    except Exception as e:
        return f"A critical error occurred: {str(e)}"


# Create and launch the Gradio interface
iface = gr.Interface(
    fn=run_research,
    inputs=gr.Textbox(
        lines=3,
        placeholder="e.g., 'Analyze Nvidia stock' or 'Compare TSLA vs RIVN'",
        label="Query",
    ),
    outputs=gr.Markdown(label="Research Report", elem_id="markdown-output"),
    title="🤖 Multi-Agent Stock Research",
    description="""
    Enter a query to research a single stock or compare two.
    The system uses multiple agents to parse, gather, analyze, and report.
    """,
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