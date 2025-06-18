import argparse
from workflow import Workflow


def main():
    """Main function to run the CLI application."""
    parser = argparse.ArgumentParser(
        description="Run the stock research agent from your terminal."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Your query, e.g., 'Compare AAPL and MSFT'",
    )
    args = parser.parse_args()

    print(f"🚀 Starting research for query: '{args.query}'")

    # Instantiate and run the workflow
    workflow = Workflow()
    final_state = workflow.run(args.query)

    print("\n" + "─" * 20)
    print("✅ Research Complete")
    print("─" * 20 + "\n")

    if final_state.get("error_messages"):
        print("❌ Errors Occurred:")
        for error in final_state["error_messages"]:
            print(f"- {error}")
    else:
        if final_state.get("executive_summary"):
            print("📝 Executive Summary:")
            print(final_state["executive_summary"])

        if final_state.get("comparison_dashboard"):
            print(final_state["comparison_dashboard"])

    print("\n" + "─" * 20)
    print("🕵️ Agent Messages")
    print("─" * 20)
    for msg in final_state.get("messages", []):
        print(f"- {msg}")


if __name__ == "__main__":
    main()