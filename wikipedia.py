# issue with tool and it's set_lang function
import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Wikipedia Search")
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information on a topic.
    
    Args:
        query: Topic or keyword to search for on Wikipedia
    Returns:
        String containing Wikipedia article content with summary and details
    """
    try:
        wikipedia_wrapper = WikipediaAPIWrapper(
            top_k_results=2,  
            doc_content_chars_max=4000  
        )
        
        wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
        result = wikipedia_tool.run(query)
        
        if not result or "No good Wikipedia Search Result was found" in result:
            return f"No Wikipedia articles found for '{query}'. Please try a different search term."
        
        return result
    
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

def create_wikipedia_researcher(llm):
    return Agent(
        role="Wikipedia Research Specialist",
        goal="Search and analyze Wikipedia articles to provide comprehensive, factual information on various topics",
        backstory=(
            "You are an expert researcher with access to Wikipedia's vast knowledge base. "
            "You excel at finding relevant information, summarizing complex topics, "
            "and providing well-structured, factual responses based on Wikipedia content."
        ),
        tools=[search_wikipedia],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_research_task(topic):
    return Task(
        description=(
            f"Research '{topic}' using Wikipedia and provide a comprehensive overview. "
            f"Include key concepts, historical background, current developments, and "
            f"important details. Organize the information in a clear, structured format."
        ),
        expected_output=(
            f"A comprehensive research report on '{topic}' including:\n"
            "- Clear definition and overview\n"
            "- Historical background and development\n"
            "- Key concepts and components\n"
            "- Current status and recent developments\n"
            "- Notable figures or organizations involved\n"
            "- Relevant applications or examples"
        ),
        agent=None
    )

def check_requirements():
    # Check API key
    if not GEMINI_API_KEY:
        print("❌ Missing GEMINI_API_KEY environment variable")
        print("Create a .env file with: GEMINI_API_KEY=your_api_key_here")
        return False
    
    # Check dependencies
    try:
        import wikipedia
        wikipedia.search("test", results=1)
    except ImportError:
        print("❌ Missing dependency: wikipedia")
        print("Install with: pip install wikipedia")
        return False
    except Exception:
        pass  # This is expected - just testing if the module imports
    
    return True

def main():
    if not check_requirements():
        return
    
    print("📚 Wikipedia Research Tool")
    print("=" * 30)
    
    try:
        # Setup AI
        llm = setup_gemini_llm()
        agent = create_wikipedia_researcher(llm)
        
        while True:
            # Get user input
            topic = input("\n🔍 Enter topic to research (or 'exit' to quit): ").strip()
            
            if topic.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not topic:
                print("❌ Please enter a topic.")
                continue
            
            print(f"\n🚀 Researching: '{topic}'...")
            print("-" * 30)
            
            # Create and run task
            task = create_research_task(topic)
            task.agent = agent
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False
            )
            
            result = crew.kickoff()
            print("\n" + "=" * 30)
            print("📖 RESEARCH RESULTS:")
            print("=" * 30)
            print(result)
            print("=" * 30)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
