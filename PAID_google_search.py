# needs Custom Search API key
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Google Search")
def search_google(query: str) -> str:
    """
    Search Google for recent results and return formatted information.
    Args:
        query: Search query (e.g., "latest AI developments", "python programming")
    Returns:
        String containing search results with titles, snippets, and links
    """
    try:
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            return "Error: GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in environment variables"
        
        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID
        )
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error searching Google: {str(e)}"

@tool("Google Search with Metadata")
def search_google_detailed(query: str, num_results: int = 5) -> str:
    """
    Search Google and return detailed results with metadata.
    Args:
        query: Search query
        num_results: Number of results to return (1-10)
    Returns:
        Formatted string with titles, snippets, and links
    """
    try:
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            return "Error: GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in environment variables"
        
        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID,
            k=min(num_results, 10)
        )
        results = search.results(query, num_results)
        
        if not results:
            return f"No results found for query: {query}"
        
        formatted_results = f"Google Search Results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. **{result.get('title', 'No Title')}**\n"
            formatted_results += f"   Link: {result.get('link', 'No Link')}\n"
            formatted_results += f"   Snippet: {result.get('snippet', 'No Description')}\n\n"
        
        return formatted_results
    except Exception as e:
        return f"Error in detailed Google search: {str(e)}"

@tool("Google Quick Search")
def quick_google_search(query: str) -> str:
    """
    Perform a quick Google search and return just the top result.
    Args:
        query: Search query
    Returns:
        Top search result with title and snippet
    """
    try:
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            return "Error: GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in environment variables"
        
        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID,
            k=1
        )
        result = search.run(query)
        return f"Top result for '{query}':\n{result}"
    except Exception as e:
        return f"Error in quick Google search: {str(e)}"

def create_search_agent(llm, tools):
    return Agent(
        role="Information Research Specialist",
        goal="Conduct comprehensive web searches to find accurate, current information and provide well-organized summaries",
        backstory="You are an expert researcher with access to Google Search, skilled at finding relevant information, verifying facts, and synthesizing data from multiple sources.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_search_task(search_request, search_params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in search_params.items()]) if search_params else "None"
    
    task_description = f"""
    You have access to multiple Google Search tools for finding information on the web.

    ### Search Request:
    {search_request}

    ### Search Parameters:
    {params_str}

    Please:
    1. Analyze the search request to determine the best search approach
    2. Use appropriate search tools based on the request type:
    - Use "Google Search" for general searches
    - Use "Google Search with Metadata" for detailed results with multiple sources
    - Use "Google Quick Search" for simple, focused queries
    3. If needed, perform multiple searches with different queries to get comprehensive results
    4. Synthesize the information from search results into a clear, organized response
    5. Include relevant links and sources in your summary

    Available tools:
    - Google Search: General web search with basic results
    - Google Search with Metadata: Detailed search with structured results and metadata
    - Google Quick Search: Fast search returning only the top result
    """
        
    return Task(
        description=task_description,
        expected_output="Comprehensive research summary with relevant information, sources, and links",
        agent=None  
    )

def get_user_input():
    print("Google Search Research Tool")
    print("=" * 30)
    print("Search types:")
    print("1. General search (multiple results)")
    print("2. Detailed search (with metadata)")
    print("3. Quick search (top result only)")
    print("4. Research query (comprehensive analysis)")
    
    choice = input("\nSelect search type (1-4): ").strip()
    search_params = {}
    
    if choice == "1":
        query = input("Enter your search query: ").strip()
        search_request = f"Perform a general web search for: {query}"
        search_params = {"query": query, "search_type": "general"}
    elif choice == "2":
        query = input("Enter your search query: ").strip()
        try:
            num_results = int(input("Number of results (1-10, default 5): ").strip() or "5")
            num_results = min(max(num_results, 1), 10)
        except ValueError:
            num_results = 5
        search_request = f"Perform a detailed search with metadata for: {query}"
        search_params = {"query": query, "num_results": num_results, "search_type": "detailed"}
    elif choice == "3":
        query = input("Enter your search query: ").strip()
        search_request = f"Perform a quick search for: {query}"
        search_params = {"query": query, "search_type": "quick"}
    elif choice == "4":
        topic = input("Enter research topic: ").strip()
        search_request = f"Conduct comprehensive research on: {topic}"
        search_params = {"topic": topic, "search_type": "research"}
    else:
        query = input("Enter your search query: ").strip() or "latest technology news"
        search_request = f"Search for: {query}"
        search_params = {"query": query, "search_type": "general"}
    
    return search_request, search_params

def main():
    # Check for required API keys
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("⚠️ Please set your Google Search API credentials:")
        print("   1. GOOGLE_API_KEY: Get from Google Cloud Console")
        print("      https://console.cloud.google.com/apis/credentials")
        print("   2. GOOGLE_CSE_ID: Get from Programmable Search Engine")
        print("      https://programmablesearchengine.google.com/controlpanel/create")
        print("   3. Add both to your .env file")
        return
    
    search_request, search_params = get_user_input()
    
    print(f"\n🚀 Starting Google Search Research")
    print(f"Request: {search_request}")
    
    tools = [
        search_google,
        search_google_detailed,
        quick_google_search
    ]
    
    llm = setup_gemini_llm()
    agent = create_search_agent(llm, tools)
    task = create_search_task(search_request, search_params)
    task.agent = agent
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("\n🔍 Conducting Google Search...")
    print("=" * 50)
    result = crew.kickoff()
    print("\n📋 Search Results:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
