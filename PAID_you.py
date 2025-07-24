import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools.you import YouSearchTool
from langchain_community.utilities.you import YouSearchAPIWrapper


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YDC_API_KEY = os.getenv("YDC_API_KEY")


def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )


@tool("You.com Web Search")
def search_you_com(query: str) -> str:
    """
    Search the web using You.com API for current information.

    Args:
        query: Search query for finding current web information
    Returns:
        String containing search results with URLs, titles, and snippets
    """
    try:
        api_wrapper = YouSearchAPIWrapper(
            ydc_api_key=YDC_API_KEY,
            num_web_results=5
        )

        you_tool = YouSearchTool(api_wrapper=api_wrapper)
        results = you_tool.invoke(query)

        if isinstance(results, list):
            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_results.append(f"""
                Result {i}:
                Title: {doc.metadata.get('title', 'No title')}
                URL: {doc.metadata.get('url', 'No URL')}
                Description: {doc.metadata.get('description', 'No description')}
                Content: {doc.page_content[:200]}...
                """)
            return "\n".join(formatted_results)
        else:
            return str(results)

    except Exception as e:
        return f"Error searching You.com: {str(e)}"


def create_web_researcher(llm):
    return Agent(
        role="Web Research Specialist",
        goal="Search the web for current, accurate information using You.com API",
        backstory=(
            "You are an expert web researcher who specializes in finding the most "
            "recent and relevant information from across the internet. You use "
            "You.com's powerful search capabilities to ground responses in factual, "
            "up-to-date data that may not be in training datasets."
        ),
        tools=[search_you_com],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_search_task(query="latest AI developments", num_results=5):
    return Task(
        description=(
            f"Search the web for information about '{query}' using You.com API. "
            f"Find the most current and relevant information available. "
            f"Analyze the search results and provide a comprehensive summary "
            f"with key insights and sources."
        ),
        expected_output=(
            f"A comprehensive report on '{query}' including:\n"
            "- Key findings from current web sources\n"
            "- Important URLs and references\n"
            "- Analysis of the most relevant information\n"
            "- Summary of current trends or developments"
        ),
        agent=None
    )


def check_dependencies():
    missing_deps = []
    try:
        from langchain_community.tools.you import YouSearchTool
        from langchain_community.utilities.you import YouSearchAPIWrapper
    except ImportError:
        missing_deps.append("langchain-community")
    return missing_deps


def main():
    if not GEMINI_API_KEY:
        print("⚠️  Please set your GEMINI_API_KEY environment variable")
        return

    if not YDC_API_KEY:
        print("⚠️  Please set your YDC_API_KEY environment variable")
        print("Get your API key from: https://you.com/")
        return

    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️  Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nTo install missing dependencies, run:")
        print("pip install langchain-community")
        return

    print("🚀 Starting You.com Web Search with Gemini 2.5 Flash...")

    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini 2.5 Flash LLM configured")

    # Create agent
    researcher = create_web_researcher(gemini_llm)
    print("✅ You.com web researcher agent created")

    # Create task
    search_task = create_search_task("CrewAI framework 2024", 5)
    search_task.agent = researcher
    print("✅ Web search task configured")
    
    # Create and run crew
    crew = Crew(
        agents=[researcher],
        tasks=[search_task],
        verbose=True
    )

    print("\n🔍 Executing You.com web search...")
    result = crew.kickoff()
    print(result)


def run():
    """Alternative entry point for crewai run command"""
    main()


if __name__ == "__main__":
    main()

