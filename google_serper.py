import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Google Serper Search")
def search_web(query: str) -> str:
    """
    Search the web using Google Serper API for current information.
    Args:
        query: Search query (e.g., "latest AI news", "Tesla stock price")
    Returns:
        String containing search results with organic results, knowledge graph, and answer box
    """
    try:
        if not SERPER_API_KEY:
            return "Error: SERPER_API_KEY not found. Please set your Serper API key in environment variables."
        
        search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
        results = search.run(query)
        return f"Search results for '{query}':\n{results}"
    except Exception as e:
        return f"Error searching with Serper: {str(e)}"

@tool("Google Serper Detailed Search")
def search_web_detailed(query: str, result_type: str = "search") -> str:
    """
    Perform detailed search with structured results using Google Serper API.
    Args:
        query: Search query
        result_type: Type of search - "search", "images", "news", or "places"
    Returns:
        Detailed formatted search results with metadata
    """
    try:
        if not SERPER_API_KEY:
            return "Error: SERPER_API_KEY not found. Please set your Serper API key in environment variables."
        
        search_type = result_type if result_type in ["search", "images", "news", "places"] else "search"
        
        search = GoogleSerperAPIWrapper(
            serper_api_key=SERPER_API_KEY,
            type=search_type
        )
        results = search.results(query)
        
        formatted_results = f"Detailed {search_type} results for '{query}':\n\n"
        
        if result_type == "search":
            if 'knowledgeGraph' in results:
                kg = results['knowledgeGraph']
                formatted_results += f"**Knowledge Graph:**\n"
                formatted_results += f"Title: {kg.get('title', 'N/A')}\n"
                formatted_results += f"Type: {kg.get('type', 'N/A')}\n"
                formatted_results += f"Description: {kg.get('description', 'N/A')}\n"
                formatted_results += f"Website: {kg.get('website', 'N/A')}\n\n"
            
            if 'organic' in results:
                formatted_results += f"**Top Organic Results:**\n"
                for i, result in enumerate(results['organic'][:5], 1):
                    formatted_results += f"{i}. **{result.get('title', 'No Title')}**\n"
                    formatted_results += f"   URL: {result.get('link', 'No Link')}\n"
                    formatted_results += f"   Snippet: {result.get('snippet', 'No Description')}\n\n"
        
        elif result_type == "news":
            if 'news' in results:
                formatted_results += f"**Latest News:**\n"
                for i, article in enumerate(results['news'][:5], 1):
                    formatted_results += f"{i}. **{article.get('title', 'No Title')}**\n"
                    formatted_results += f"   Source: {article.get('source', 'Unknown')}\n"
                    formatted_results += f"   Date: {article.get('date', 'Unknown')}\n"
                    formatted_results += f"   URL: {article.get('link', 'No Link')}\n"
                    formatted_results += f"   Snippet: {article.get('snippet', 'No Description')}\n\n"
        
        elif result_type == "images":
            if 'images' in results:
                formatted_results += f"**Image Results:**\n"
                for i, image in enumerate(results['images'][:5], 1):
                    formatted_results += f"{i}. **{image.get('title', 'No Title')}**\n"
                    formatted_results += f"   Image URL: {image.get('imageUrl', 'No URL')}\n"
                    formatted_results += f"   Source: {image.get('source', 'Unknown')}\n"
                    formatted_results += f"   Link: {image.get('link', 'No Link')}\n\n"
        
        elif result_type == "places":
            if 'places' in results:
                formatted_results += f"**Places Results:**\n"
                for i, place in enumerate(results['places'][:5], 1):
                    formatted_results += f"{i}. **{place.get('title', 'No Title')}**\n"
                    formatted_results += f"   Address: {place.get('address', 'No Address')}\n"
                    formatted_results += f"   Rating: {place.get('rating', 'No Rating')} ({place.get('ratingCount', '0')} reviews)\n"
                    formatted_results += f"   Phone: {place.get('phoneNumber', 'No Phone')}\n"
                    formatted_results += f"   Website: {place.get('website', 'No Website')}\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error in detailed Serper search: {str(e)}"

@tool("Google Serper News Search")
def search_news(query: str, time_filter: str = "") -> str:
    """
    Search for recent news using Google Serper API.
    Args:
        query: News search query
        time_filter: Time filter - "qdr:h" (hour), "qdr:d" (day), "qdr:w" (week), "qdr:m" (month)
    Returns:
        Recent news articles related to the query
    """
    try:
        if not SERPER_API_KEY:
            return "Error: SERPER_API_KEY not found. Please set your Serper API key in environment variables."
        
        search_params = {
            "serper_api_key": SERPER_API_KEY,
            "type": "news"
        }
        
        if time_filter:
            search_params["tbs"] = time_filter
        
        search = GoogleSerperAPIWrapper(**search_params)
        results = search.results(query)
        
        if 'news' not in results or not results['news']:
            return f"No recent news found for '{query}'"
        
        formatted_results = f"Recent news for '{query}':\n\n"
        for i, article in enumerate(results['news'][:7], 1):
            formatted_results += f"{i}. **{article.get('title', 'No Title')}**\n"
            formatted_results += f"   Source: {article.get('source', 'Unknown')}\n"
            formatted_results += f"   Published: {article.get('date', 'Unknown')}\n"
            formatted_results += f"   Summary: {article.get('snippet', 'No Summary')}\n"
            formatted_results += f"   URL: {article.get('link', 'No Link')}\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error searching news with Serper: {str(e)}"


def create_serper_agent(llm, tools):
    return Agent(
        role="Web Research Specialist",
        goal="Conduct comprehensive web searches using Google Serper to find accurate, current information and provide well-organized summaries",
        backstory="You are an expert web researcher with access to Google Serper API, skilled at finding relevant information, current news, and comprehensive data from across the internet.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_serper_task(search_request, search_params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in search_params.items()]) if search_params else "None"
    
    task_description = f"""
    You have access to multiple Google Serper search tools for finding information on the web.

    ### Search Request:
    {search_request}

    ### Search Parameters:
    {params_str}

    Please:
    1. Analyze the search request to determine the best search approach
    2. Use appropriate search tools based on the request type:
    - Use "Google Serper Search" for general web searches
    - Use "Google Serper Detailed Search" for comprehensive results with metadata
    - Use "Google Serper News Search" for current news and recent events
    3. If needed, perform multiple searches with different queries to get comprehensive results
    4. Synthesize the information from search results into a clear, organized response
    5. Include relevant sources and links in your summary

    Available tools:
    - Google Serper Search: Fast general web search
    - Google Serper Detailed Search: Comprehensive search with knowledge graph, organic results, images, news, or places
    - Google Serper News Search: Focused news search with time filtering options
    """
    
    return Task(
        description=task_description,
        expected_output="Comprehensive research summary with relevant information, sources, and links from current web data",
        agent=None  # to be assigned later
    )

def get_user_input():
    print("Google Serper Web Search Tool")
    print("=" * 32)
    print("Search types:")
    print("1. General web search")
    print("2. Detailed search (with knowledge graph)")
    print("3. News search")
    print("4. Image search")
    print("5. Places search")
    print("6. Research query (comprehensive)")
    
    choice = input("\nSelect search type (1-6): ").strip()
    search_params = {}
    
    if choice == "1":
        query = input("Enter your search query: ").strip()
        search_request = f"Perform a general web search for: {query}"
        search_params = {"query": query, "search_type": "general"}
    elif choice == "2":
        query = input("Enter your search query: ").strip()
        search_request = f"Perform a detailed search with metadata for: {query}"
        search_params = {"query": query, "result_type": "search", "search_type": "detailed"}
    elif choice == "3":
        query = input("Enter news search query: ").strip()
        print("Time filters: h (hour), d (day), w (week), m (month)")
        time_filter = input("Enter time filter (optional): ").strip()
        if time_filter and time_filter in ['h', 'd', 'w', 'm']:
            time_filter = f"qdr:{time_filter}"
        search_request = f"Search for recent news about: {query}"
        search_params = {"query": query, "time_filter": time_filter, "search_type": "news"}
    elif choice == "4":
        query = input("Enter image search query: ").strip()
        search_request = f"Search for images related to: {query}"
        search_params = {"query": query, "result_type": "images", "search_type": "images"}
    elif choice == "5":
        query = input("Enter places search query: ").strip()
        search_request = f"Search for places related to: {query}"
        search_params = {"query": query, "result_type": "places", "search_type": "places"}
    elif choice == "6":
        topic = input("Enter research topic: ").strip()
        search_request = f"Conduct comprehensive research on: {topic}"
        search_params = {"topic": topic, "search_type": "research"}
    else:
        query = input("Enter your search query: ").strip() or "latest technology news"
        search_request = f"Search for: {query}"
        search_params = {"query": query, "search_type": "general"}
    
    return search_request, search_params

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    if not SERPER_API_KEY:
        print("⚠️ Please set your SERPER_API_KEY environment variable")
        print("   1. Go to https://serper.dev")
        return
    
    search_request, search_params = get_user_input()
    
    print(f"\n🚀 Starting Google Serper Web Search")
    print(f"Request: {search_request}")
    
    tools = [
        search_web,
        search_web_detailed,
        search_news
    ]
    
    llm = setup_gemini_llm()
    agent = create_serper_agent(llm, tools)
    task = create_serper_task(search_request, search_params)
    task.agent = agent
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("\n🔍 Conducting Google Serper search...")
    print("=" * 50)
    result = crew.kickoff()
    print("\n📋 Search Results:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
