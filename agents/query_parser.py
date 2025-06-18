from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
import json
import re


class QueryParserAgent:
    """AI-powered query parsing"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0
        )

    def __call__(self, state: dict) -> dict:
        """Parse query using AI"""
        # Work with dictionary state
        query_text = state["query"]["user_input"]

        prompt = f"""
        Extract stock information from this query: "{query_text}"

        Rules:
        - Convert company names to tickers (e.g., "Apple" → "AAPL", "Microsoft" → "MSFT")
        - Identify if user wants comparison (words like: compare, vs, versus, or, better)
        - Maximum 2 stocks for comparison

        Return ONLY a valid JSON object with this exact format:
        {{
            "tickers": ["TICKER1", "TICKER2"],
            "company_names": ["Name1", "Name2"],
            "is_comparison": true,
            "query_intent": "what user wants to know"
        }}
        """

        try:
            response = self.llm.invoke(prompt)

            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)

            # Debug print
            print(f"LLM Response: {content}")

            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole content
                parsed = json.loads(content)

            tickers = parsed.get("tickers", [])

            # Fallback: try regex if AI didn't find tickers
            if not tickers:
                tickers = re.findall(r'\b[A-Z]{1,5}\b', query_text)
                tickers = [t for t in tickers if t not in {'I', 'A', 'AND', 'OR', 'VS'}]

            if not tickers:
                state["error_messages"].append("Could not identify any stocks in your query")
                return state

            # Determine analysis type
            is_comparison = parsed.get("is_comparison", False) or len(tickers) > 1
            analysis_type = "comparison" if is_comparison else "single"

            state["query"]["tickers"] = tickers[:2]  # Max 2
            state["query"]["analysis_type"] = analysis_type

            state["messages"].append(
                f"Analyzing: {', '.join(tickers)} "
                f"({', '.join(parsed.get('company_names', tickers))})"
            )

        except json.JSONDecodeError as e:
            # More detailed error message
            state["error_messages"].append(f"Failed to parse LLM response as JSON: {str(e)}")

            # Try fallback regex parsing
            tickers = re.findall(r'\b[A-Z]{1,5}\b', query_text)
            tickers = [t for t in tickers if t not in {'I', 'A', 'AND', 'OR', 'VS', 'AAPL'}] #TODO

            if tickers:
                state["query"]["tickers"] = list(set(tickers))[:2]
                state["query"]["analysis_type"] = "comparison" if len(tickers) > 1 else "single"
                state["messages"].append(f"Fallback parsing found: {', '.join(tickers)}")
                # Clear the error since we recovered
                state["error_messages"] = []

        except Exception as e:
            state["error_messages"].append(f"Query parsing error: {str(e)}")

        return state