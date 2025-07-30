import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

from langchain_agentql.tools import ExtractWebDataTool

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AGENTQL_API_KEY = os.getenv("AGENTQL_API_KEY")

# === LLM SETUP ===
def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

# === AGENTQL TOOL WRAPPER ===
@tool("AgentQL Web Data Extractor")
def agentql_extract_web_data(
    url: str,
    prompt: str = "",
    query: str = "",
    mode: str = "fast",
    is_stealth_mode_enabled: bool = False,
    is_scroll_to_bottom_enabled: bool = False,
    timeout: int = 120
) -> str:
    """
    Extracts structured data as JSON from a web page using AgentQL API.
    Args:
        url: The target web page URL.
        prompt: A natural language description of data to extract (optional).
        query: AgentQL query (optional, overrides prompt if both set).
        mode: "fast" (default) or "standard".
        is_stealth_mode_enabled: Enable anti-bot evasion (default: False).
        is_scroll_to_bottom_enabled: Scroll before extraction (default: False).
        timeout: Request timeout in seconds.
    Returns:
        Extracted web data JSON or error.
    """
    try:
        tool = ExtractWebDataTool(
            api_key=AGENTQL_API_KEY,
            timeout=timeout,
            is_stealth_mode_enabled=is_stealth_mode_enabled,
            is_scroll_to_bottom_enabled=is_scroll_to_bottom_enabled,
            mode=mode
        )
        params = {"url": url}
        if query:
            params["query"] = query
        elif prompt:
            params["prompt"] = prompt
        results = tool.invoke(params)
        return f"AgentQL Extraction Results for '{url}':\n{results}"
    except Exception as e:
        return f"Error extracting web data: {str(e)}"

# === AGENT ===
def create_agentql_agent(llm, tools):
    return Agent(
        role="Web Data Extraction Specialist",
        goal="Extract structured web data using AgentQL from any page with AgentQL queries or natural language prompts.",
        backstory="You are an expert in robust web scraping and AI data extraction, using AgentQL tools to handle modern, dynamic sites.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# === TASK CREATION ===
def create_agentql_task(request, params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in params.items()]) if params else "None"
    task_description = f"""
You have access to AgentQL tools for robust web data extraction.

### Extraction Request:
{request}

### Parameters:
{params_str}

Please use "AgentQL Web Data Extractor" to extract structured data (JSON) from the provided URL, using either a natural language prompt or a specific AgentQL query. If a query is provided, use it directly. Otherwise, use the prompt as a natural language instruction.
"""
    return Task(
        description=task_description,
        expected_output="Structured JSON web data or an error message.",
        agent=None
    )

# === USER INPUT ===
def get_user_input():
    print("AgentQL Web Extraction Tool")
    print("=" * 32)
    url = input("Web page URL: ").strip()
    prompt = input("Describe what to extract (leave blank to use AgentQL query): ").strip()
    query = ""
    if not prompt:
        query = input("AgentQL query (e.g. { posts[] { title url date author } }): ").strip()
    mode = input("Mode (fast/standard, default fast): ").strip() or "fast"
    is_stealth = input("Enable stealth mode? (y/N): ").strip().lower() == 'y'
    is_scroll = input("Scroll to bottom before extraction? (y/N): ").strip().lower() == 'y'
    timeout = int(input("Timeout (seconds, default 120): ").strip() or "120")
    extraction_request = f"Extract data from {url} using {'AgentQL query' if query else 'natural language prompt'}"
    extraction_params = {
        "url": url, "prompt": prompt, "query": query, "mode": mode,
        "is_stealth_mode_enabled": is_stealth,
        "is_scroll_to_bottom_enabled": is_scroll,
        "timeout": timeout
    }
    return extraction_request, extraction_params

# === MAIN EXECUTION ===
def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    if not AGENTQL_API_KEY:
        print("⚠️ Please set your AGENTQL_API_KEY environment variable")
        print("  Get it from https://dev.agentql.com/api-keys")
        return
    extraction_request, extraction_params = get_user_input()
    print(f"\n🚀 Starting AgentQL Extraction: {extraction_request}")
    tools = [agentql_extract_web_data]
    llm = setup_gemini_llm()
    agent = create_agentql_agent(llm, tools)
    task = create_agentql_task(extraction_request, extraction_params)
    task.agent = agent

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    print("\n🕷️ Running AgentQL web extraction...\n" + "="*50)
    result = crew.kickoff()
    print("\n📝 Extraction Result:\n" + "="*50)
    print(result)

if __name__ == "__main__":
    main()
