# Ollama Search Agent - MCP Folder

This folder contains scripts related to the MCP (Model Context Protocol) server and client for the Ollama Search Agent project.

## Scripts

### mcp_server.py
- **Purpose:** Starts and manages the MCP server.
- **Usage:** Run this script to launch the MCP server, which handles requests and serves responses using the configured model and tools.
- **Main Features:**
  - Initializes the MCP server
  - Configures transport and port
  - Integrates with LangChain and Ollama models

### mcp_research_client.py
- **Purpose:** Acts as a client to interact with the MCP server for research and testing purposes.
- **Usage:** Use this script to send queries to the MCP server and receive responses, useful for development and debugging.
- **Main Features:**
  - Sends requests to the MCP server
  - Prints and analyzes responses
  - Demonstrates tool and AI message extraction

## How to Run the Scripts

1. **Activate your Python environment:**
   ```bash
   source ../myenv/bin/activate
   ```
   (Adjust the path if your environment is elsewhere.)

2. **Run the MCP server:**
   ```bash
   python mcp_server.py
   ```
   This will start the MCP server using the default configuration.

3. **Run the research client:**
   ```bash
   python mcp_research_client.py
   ```
   This will send a test query to the MCP server and print the response.

**Note:** Make sure the MCP server is running before starting the client.

## Additional Notes
- Both scripts rely on the LangChain and Ollama libraries for AI and tool integration.
- Ensure the required dependencies are installed (see `requirements.txt`).
- Scripts are designed to be run from the command line using the configured Python environment.

---
For more details, refer to the inline comments in each script or the main project README.
