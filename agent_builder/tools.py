from langchain_core.tools import tool

@tool
def read_file(file_path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Writes content to a file."""
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def code_analysis(code_snippet: str) -> str:
    """Analyzes the given code snippet for potential issues."""
    return f"Analysis complete: The code seems structurally sound but lacks comments. (Mock Analysis for: {code_snippet[:20]}...)"

@tool
def web_search(query: str) -> str:
    """Performs a web search."""
    return f"Search results for '{query}': [Mock Result 1], [Mock Result 2]"

# List of all available tools
ALL_TOOLS = [read_file, write_file, code_analysis, web_search]
