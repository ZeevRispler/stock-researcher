from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from config import TAVILY_API_KEY, OPENAI_API_KEY, OPENAI_API_BASE

search_tool = TavilySearch(max_results=7, tavily_api_key=TAVILY_API_KEY)
tools = [search_tool]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0.3,
)

react_agent = create_react_agent(llm, tools)


class MarketDataAgent:
    def __call__(self, state: dict) -> dict:
        print("📊 Fetching market data...")

        tickers = state["query"].get("tickers", [])
        if not tickers:
            state["error_messages"].append("No tickers found to analyze")
            return state

        for ticker in tickers:
            prompt = f"""
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
                response = react_agent.invoke({"messages": [{"role": "user", "content": prompt}]})

                if isinstance(response, dict):
                    output_text = response.get("output", "") or response.get("content", "") or str(response)
                    if "messages" in response and response["messages"]:
                        last_message = response["messages"][-1]
                        if isinstance(last_message, dict):
                            output_text = last_message.get("content", str(last_message))
                        else:
                            output_text = str(last_message)
                else:
                    output_text = str(response)

                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["market_data"] = output_text

            except Exception as e:
                error_message = f"Error running ReAct agent for {ticker}: {str(e)}"
                state["error_messages"].append(error_message)

                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["market_data"] = f"Failed to fetch data for {ticker}"

        return state