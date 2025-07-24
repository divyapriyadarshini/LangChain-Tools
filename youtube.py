import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import YouTubeSearchTool

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("YouTube Video Search")
def search_youtube_videos(query: str) -> str:
    """
    Args:
        query: Search query in format 'topic, max_results' or just 'topic'
    Returns:
        String containing YouTube video search results
    """
    try:
        youtube_tool = YouTubeSearchTool()
        return youtube_tool.run(query)
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"


def create_youtube_researcher(llm):
    return Agent(
        role="YouTube URL Finder",
        goal="Find and return YouTube video URLs for specific topics or personalities",
        backstory=(
            "You are a specialized search agent focused on finding YouTube videos. "
            "Your job is to locate relevant video URLs and present them in a clean, "
            "organized format without attempting to analyze content you cannot access."
        ),
        tools=[search_youtube_videos],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_search_task(query="lex fridman", max_results=5):
    return Task(
        description=(
            f"Search for YouTube videos about '{query}' and return the URLs. "
            f"Use the format '{query}, {max_results}' when calling the YouTube search tool. "
            f"Present the results as a clean list of URLs only."
        ),
        expected_output=(
            f"A clean list of the top {max_results} YouTube video URLs for '{query}', "
            "formatted as:\n"
            "1. https://www.youtube.com/watch?v=...\n"
            "2. https://www.youtube.com/watch?v=...\n"
            "etc."
        ),
        agent=None
    )


def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return

    print("🚀 Starting CrewAI YouTube Search ")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini LLM configured")

    # Create agent
    researcher = create_youtube_researcher(gemini_llm)
    print("✅ YouTube researcher agent created")

    # Create task
    search_task = create_search_task("lex fridman", 7)
    search_task.agent = researcher
    print("✅ Search task configured")

    # Create and run crew
    crew = Crew(
        agents=[researcher],
        tasks=[search_task],
        verbose=True
    )

    print("\nExecuting YouTube search...")
    print("=" * 50)
    result = crew.kickoff()
    print(result)

def run():
    """Alternative entry point for crewai run command"""
    main()


if __name__ == "__main__":
    main()