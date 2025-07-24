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

def create_search_task(query, max_results):
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

def get_user_input():
    """Get search topic and number of results from user"""
    print("YouTube Video Search Tool")
    print("=" * 30)
    
    # Get search topic
    topic = input("Enter the topic to search for: ").strip()
    if not topic:
        print("⚠️ Topic cannot be empty. Using default: 'lex fridman'")
        topic = "lex fridman"
    
    # Get number of results
    while True:
        try:
            num_results = input("Enter number of videos to find (default 5): ").strip()
            if not num_results:
                num_results = 5
                break
            num_results = int(num_results)
            if num_results <= 0:
                print("⚠️ Please enter a positive number")
                continue
            if num_results > 20:
                print("⚠️ Maximum 20 results allowed. Setting to 20.")
                num_results = 20
            break
        except ValueError:
            print("⚠️ Please enter a valid number")
    
    return topic, num_results

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return

    # Get user input
    topic, max_results = get_user_input()
    
    print(f"\n🚀 Starting CrewAI YouTube Search for '{topic}' ({max_results} results)")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini LLM configured")

    # Create agent
    researcher = create_youtube_researcher(gemini_llm)
    print("✅ YouTube researcher agent created")

    # Create task with user input
    search_task = create_search_task(topic, max_results)
    search_task.agent = researcher
    print("✅ Search task configured")

    # Create and run crew
    crew = Crew(
        agents=[researcher],
        tasks=[search_task],
        verbose=True
    )

    print(f"\nExecuting YouTube search for '{topic}'...")
    print("=" * 50)
    result = crew.kickoff()
    print(result)

def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
