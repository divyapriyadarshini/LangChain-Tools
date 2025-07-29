import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.utilities import GoogleScholarAPIWrapper

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Google Scholar Search")
def search_google_scholar(query: str) -> str:
    """
    Args:
        query: Search query for academic papers (e.g., "machine learning", "climate change")
    Returns:
        String containing formatted search results with titles, authors, summaries, and citation counts
    """
    try:
        if not SERP_API_KEY:
            return "Error: SERP_API_KEY not found. Please set your SerpAPI key in the environment variables."
        
        scholar = GoogleScholarAPIWrapper(serp_api_key=SERP_API_KEY)
        results = scholar.run(query)
        return results
    except Exception as e:
        return f"Error searching Google Scholar: {str(e)}"


def create_scholar_agent(llm):
    return Agent(
        role="Academic Research Specialist",
        goal="Conduct comprehensive literature searches and provide structured summaries of academic research",
        backstory="You are an expert academic researcher with access to Google Scholar database, skilled at finding relevant papers and analyzing research trends.",
        tools=[search_google_scholar],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_research_task(topic):
    return Task(
        description=f"""
        Conduct a comprehensive literature search on '{topic}' using Google Scholar.
        Find the most relevant and highly-cited academic papers and provide a structured summary.
        
        Your analysis should include:
        1. Top 5-10 most relevant papers
        2. Key research themes and trends
        3. Notable authors and institutions
        4. Citation counts and impact
        5. Recent developments (if available)
        
        Format the results in a clear, academic style with proper citations.
        """,
        expected_output="Structured academic literature review with paper summaries, key findings, and research trends",
        agent=None
    )


def get_user_input():
    print("Google Scholar Academic Research")
    print("===================================")
    topic = input("Enter the academic topic to research: ").strip() or "stem cells"
    return topic


def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    if not SERP_API_KEY:
        print("⚠️ Please set your SERP_API_KEY environment variable")
        print("   1. Go to https://serpapi.com")
        print("   2. Sign up for a free account")
        print("   3. Get your API key")
        print("   4. Add SERP_API_KEY=your_key_here to your .env file")
        return

    topic = get_user_input()
    
    print(f"\n🚀 Starting Google Scholar Research for '{topic}'")
    
    llm = setup_gemini_llm()
    print("✅ Gemini LLM configured")
    
    agent = create_scholar_agent(llm)
    print("✅ Academic researcher agent created")
    
    task = create_research_task(topic)
    task.agent = agent
    print("✅ Research task configured")
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print(f"\nExecuting Google Scholar research for '{topic}'...")
    print("=" * 50)
    
    result = crew.kickoff()
    print("\n📚 Research Results:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
