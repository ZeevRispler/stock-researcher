import argparse
from workflow import Workflow


def main():
    parser = argparse.ArgumentParser(description="Run the stock research agent from your terminal.")
    parser.add_argument("query", type=str, help="Your query, e.g., 'Compare AAPL and MSFT'")
    args = parser.parse_args()

    workflow = Workflow()
    final_state = workflow.run(args.query)

    print("\n" + "─" * 50)

    if final_state.get("error_messages"):
        print("❌ Errors:")
        for error in final_state["error_messages"]:
            print(f"  {error}")
    else:
        if final_state.get("executive_summary"):
            print("📝 Executive Summary:")
            print(final_state["executive_summary"])

        if final_state.get("comparison_dashboard"):
            print(final_state["comparison_dashboard"])


if __name__ == "__main__":
    main()