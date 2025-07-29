import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_apify import ApifyActorsTool

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# === LLM SETUP ===
def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

# === APIFY ACTORS TOOL WRAPPERS ===
@tool("Apify RAG Web Browser")
def rag_web_browser(query: str, max_results: int = 3) -> str:
    """
    Use Apify RAG Web Browser to search and extract web content for AI applications.
    Args:
        query: Search query or topic to research
        max_results: Maximum number of results to return (1-10)
    Returns:
        Structured web content in markdown format for LLM processing
    """
    try:
        if not APIFY_API_TOKEN:
            return "Error: APIFY_API_TOKEN not found. Please set your Apify API token in environment variables."
        
        tool = ApifyActorsTool("apify/rag-web-browser")
        results = tool.invoke({
            "run_input": {
                "query": query,
                "maxResults": min(max_results, 10),
                "outputFormats": ["markdown"]
            }
        })
        return f"RAG Web Browser results for '{query}':\n{results}"
    except Exception as e:
        return f"Error using RAG Web Browser: {str(e)}"

@tool("Apify Website Content Crawler")
def website_content_crawler(start_url: str, max_pages: int = 10) -> str:
    """
    Crawl and extract content from websites using Apify Website Content Crawler.
    Args:
        start_url: Starting URL to crawl
        max_pages: Maximum number of pages to crawl (1-100)
    Returns:
        Extracted text content from the website pages
    """
    try:
        if not APIFY_API_TOKEN:
            return "Error: APIFY_API_TOKEN not found. Please set your Apify API token in environment variables."
        
        tool = ApifyActorsTool("apify/website-content-crawler")
        results = tool.invoke({
            "run_input": {
                "startUrls": [{"url": start_url}],
                "maxCrawlPages": min(max_pages, 100),
                "crawlerType": "cheerio"
            }
        })
        return f"Website content from '{start_url}':\n{results}"
    except Exception as e:
        return f"Error crawling website: {str(e)}"

@tool("Apify Google Search Scraper")
def google_search_scraper(query: str, max_results: int = 10) -> str:
    """
    Scrape Google search results using Apify Google Search Scraper.
    Args:
        query: Search query
        max_results: Maximum number of search results to return (1-100)
    Returns:
        Google search results with titles, snippets, and URLs
    """
    try:
        if not APIFY_API_TOKEN:
            return "Error: APIFY_API_TOKEN not found. Please set your Apify API token in environment variables."
        
        tool = ApifyActorsTool("apify/google-search-scraper")
        results = tool.invoke({
            "run_input": {
                "queries": [query],
                "maxPagesPerQuery": 1,
                "resultsPerPage": min(max_results, 100)
            }
        })
        return f"Google search results for '{query}':\n{results}"
    except Exception as e:
        return f"Error scraping Google search: {str(e)}"

@tool("Apify Custom Actor")
def custom_apify_actor(actor_id: str, run_input: dict) -> str:
    """
    Run any custom Apify Actor with specified parameters.
    Args:
        actor_id: Apify Actor ID (e.g., "username/actor-name")
        run_input: Dictionary containing the Actor's run input parameters
    Returns:
        Results from the custom Actor execution
    """
    try:
        if not APIFY_API_TOKEN:
            return "Error: APIFY_API_TOKEN not found. Please set your Apify API token in environment variables."
        
        tool = ApifyActorsTool(actor_id)
        results = tool.invoke({"run_input": run_input})
        return f"Custom Actor '{actor_id}' results:\n{results}"
    except Exception as e:
        return f"Error running custom Actor: {str(e)}"

# === AGENT ===
def create_apify_agent(llm, tools):
    return Agent(
        role="Web Scraping and Data Extraction Specialist",
        goal="Use Apify Actors to perform web scraping, crawling, and data extraction tasks to gather structured information from websites and search engines",
        backstory="You are an expert in web automation and data extraction with access to Apify's cloud platform featuring over 1000+ ready-made Actors for various scraping, crawling, and extraction use cases.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# === TASK CREATION ===
def create_apify_task(scraping_request, scraping_params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in scraping_params.items()]) if scraping_params else "None"
    
    task_description = f"""
You have access to Apify Actors for comprehensive web scraping and data extraction.

### Scraping Request:
{scraping_request}

### Parameters:
{scraping_params}

Please:
1. Analyze the scraping request to determine the best Actor approach
2. Use appropriate Apify tools based on the request type:
   - Use "Apify RAG Web Browser" for AI-optimized web browsing and content extraction
   - Use "Apify Website Content Crawler" for deep website crawling and content extraction
   - Use "Apify Google Search Scraper" for extracting Google search results
   - Use "Apify Custom Actor" for specialized scraping tasks with specific Actors
3. If needed, combine multiple Actors to get comprehensive data coverage
4. Process and structure the extracted data for analysis
5. Provide insights on data quality, patterns, and actionable findings

Available tools:
- Apify RAG Web Browser: AI-enhanced web browsing for LLM applications
- Apify Website Content Crawler: Deep website crawling for documentation and content
- Apify Google Search Scraper: Extract structured Google search results
- Apify Custom Actor: Run any Actor from Apify Store with custom parameters

Data formats supported: JSON, CSV, Excel, Markdown
"""
    
    return Task(
        description=task_description,
        expected_output="Comprehensive data extraction with structured results, insights, and actionable findings",
        agent=None  # to be assigned later
    )

# === USER INPUT ===
def get_user_input():
    print("Apify Actors Web Scraping Tool")
    print("=" * 35)
    print("Scraping operations:")
    print("1. RAG Web Browser (AI-optimized web search)")
    print("2. Website Content Crawler (deep site crawling)")
    print("3. Google Search Scraper (search results extraction)")
    print("4. Custom Actor (specify any Apify Actor)")
    print("5. Multi-source data extraction")
    
    choice = input("\nSelect operation (1-5): ").strip()
    scraping_params = {}
    
    if choice == "1":
        query = input("Enter search query: ").strip()
        try:
            max_results = int(input("Maximum results (1-10, default 3): ").strip() or "3")
            max_results = min(max(max_results, 1), 10)
        except ValueError:
            max_results = 3
        scraping_request = f"Use RAG Web Browser to search for: {query}"
        scraping_params = {"query": query, "max_results": max_results, "operation": "rag_browser"}
    
    elif choice == "2":
        start_url = input("Enter website URL to crawl: ").strip()
        try:
            max_pages = int(input("Maximum pages to crawl (1-100, default 10): ").strip() or "10")
            max_pages = min(max(max_pages, 1), 100)
        except ValueError:
            max_pages = 10
        scraping_request = f"Crawl website content from: {start_url}"
        scraping_params = {"start_url": start_url, "max_pages": max_pages, "operation": "website_crawler"}
    
    elif choice == "3":
        query = input("Enter Google search query: ").strip()
        try:
            max_results = int(input("Maximum search results (1-100, default 10): ").strip() or "10")
            max_results = min(max(max_results, 1), 100)
        except ValueError:
            max_results = 10
        scraping_request = f"Extract Google search results for: {query}"
        scraping_params = {"query": query, "max_results": max_results, "operation": "google_search"}
    
    elif choice == "4":
        actor_id = input("Enter Apify Actor ID (e.g., username/actor-name): ").strip()
        print("Enter run input parameters as key=value pairs (press Enter when done):")
        run_input = {}
        while True:
            param = input("Parameter (key=value or Enter to finish): ").strip()
            if not param:
                break
            try:
                key, value = param.split('=', 1)
                # Try to convert to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                run_input[key.strip()] = value
            except ValueError:
                print("Invalid format. Use key=value")
        
        scraping_request = f"Run custom Actor: {actor_id}"
        scraping_params = {"actor_id": actor_id, "run_input": run_input, "operation": "custom_actor"}
    
    elif choice == "5":
        topic = input("Enter topic for multi-source extraction: ").strip()
        scraping_request = f"Perform multi-source data extraction on: {topic}"
        scraping_params = {"topic": topic, "operation": "multi_source"}
    
    else:
        query = input("Enter scraping request: ").strip() or "latest technology news"
        scraping_request = f"Scrape data for: {query}"
        scraping_params = {"query": query, "operation": "general"}
    
    return scraping_request, scraping_params

# === MAIN EXECUTION ===
def main():
    # Check for required API keys
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    if not APIFY_API_TOKEN:
        print("⚠️ Please set your APIFY_API_TOKEN environment variable")
        print("   1. Go to https://apify.com")
        print("   2. Create a free account")
        print("   3. Get your API token from the console")
        print("   4. Add APIFY_API_TOKEN=your_token_here to your .env file")
        return
    
    # Get user input
    scraping_request, scraping_params = get_user_input()
    
    print(f"\n🚀 Starting Apify Web Scraping")
    print(f"Request: {scraping_request}")
    
    # Setup tools
    tools = [
        rag_web_browser,
        website_content_crawler,
        google_search_scraper,
        custom_apify_actor
    ]
    
    # Setup agent and task
    llm = setup_gemini_llm()
    agent = create_apify_agent(llm, tools)
    task = create_apify_task(scraping_request, scraping_params)
    task.agent = agent
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("\n🕷️ Conducting web scraping with Apify...")
    print("=" * 50)
    result = crew.kickoff()
    print("\n📊 Scraping Results:")
    print("=" * 50)
    print(result)

# === CLI ENTRY POINT ===
if __name__ == "__main__":
    main()
