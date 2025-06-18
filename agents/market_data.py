from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from config import TAVILY_API_KEY, OPENAI_API_KEY, OPENAI_API_BASE

# 1. Define the tool(s) the agent can use.
# The agent has access to the Tavily search engine.
search_tool = TavilySearch(
    max_results=7, tavily_api_key=TAVILY_API_KEY
)
tools = [search_tool]

# Create the LLM instance that will power the agent
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0,
)

# Create the agent by combining the LLM, tools, and prompt
react_agent = create_react_agent(llm, tools)

class MarketDataAgent:
    """
    A ReAct-based agent that uses the Tavily search tool to dynamically
    gather comprehensive market data for a given stock ticker.
    """

    def __call__(self, state: dict) -> dict:
        state["messages"].append("Fetching market data with ReAct agent...")

        for ticker in state["query"]["tickers"]:
            state["messages"].append(f"-> Running ReAct agent for {ticker}...")

            # Define the detailed goal for the ReAct agent.
            # It is prompted to produce a single text block that downstream
            # agents can process, maintaining compatibility.
            prompt = f"""
            You are a market data analysis agent, breaking down complex queries and 
            Gather comprehensive, up-to-date market data for the stock with ticker {ticker}.
            You must find and include the following information in your final answer:
            1.  Current stock price.
            2.  Market capitalization.
            3.  Price-to-Earnings (P/E) ratio.
            4.  The stock's Beta value.
            5.  A summary of at least 3-4 key recent news articles or events.
            6.  A summary of the company's primary business and revenue streams.

            Synthesize all of this information into a single, well-formatted text block.
            Your final answer should be just this text block, not a JSON object, as it
            will be passed to other agents for analysis.
            """

            try:
                # Invoke the agent executor to run the ReAct loop.
                response = react_agent.invoke({"input": prompt})
                output_text = response["output"]

                # Update the state with the collected data
                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["market_data"] = output_text
                state["messages"].append(
                    f"-> Successfully gathered market data for {ticker}."
                )

            except Exception as e:
                error_message = f"Error running ReAct agent for {ticker}: {str(e)}"
                state["error_messages"].append(error_message)
                state["messages"].append(error_message)

        return state