# tool_server.py

from mcp.server.fastmcp import FastMCP
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import wikipedia
import logging
import signal
import sys
import asyncio
import atexit

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

# Graceful shutdown handlers
def cleanup():
    """Cleanup function called on exit."""
    logging.info("🧹 Cleaning up resources...")
    # Add any cleanup code here (close connections, save state, etc.)

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logging.info(f"🛑 Received signal {sig}. Shutting down gracefully...")
    cleanup()
    sys.exit(0)

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    # Handle Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Handle termination signal (SIGTERM)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function to run on normal exit
    atexit.register(cleanup)
    
    logging.info("✅ Signal handlers set up. Use Ctrl+C or kill command to stop gracefully.")

# Run the server using streamable-http
if __name__ == "__main__":
    try:
        setup_signal_handlers()
        logging.info("🚀 Starting MCP Research Tools server...")
        logging.info("📡 Server running on streamable-http transport")
        logging.info("🔍 Available tools: duckduckgo_search, wikipedia_search")
        logging.info("⏹️  Press Ctrl+C to stop the server gracefully")
        
        mcp.run(transport="streamable-http")
        
    except KeyboardInterrupt:
        logging.info("🛑 KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logging.error(f"❌ Server error: {str(e)}")
    finally:
        logging.info("👋 Server shutdown complete.")
        cleanup()