# tool_server.py

from mcp.server.fastmcp import FastMCP
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import wikipedia
import logging

logging.basicConfig(level=logging.INFO)

# Initialize FastMCP server with a service name
mcp = FastMCP("ResearchTools")

# DuckDuckGo search tool
search = DuckDuckGoSearchAPIWrapper()

@mcp.tool()
async def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    logging.info(f" **** Called duckduckgo_search with: {query}")
    try:
        results = search.run(query)
        if results:
            return results  # Return the first result
        else:
            return "No results found."
    except Exception as e:
        logging.error(f"Error occurred in duckduckgo_search: {str(e)}")
        return f"Error: {str(e)}"

# Wikipedia search tool
@mcp.tool()
async def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information."""
    logging.info(f" ***** Called wikipedia_search with: {query}")
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        logging.error(f"Error occurred in wikipedia_search: {str(e)}")
        return f"Error: {str(e)}"

# Run the server using streamable-http
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
