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
            return f"No Wikipedia articles found for '{query}'. Please try a different search term or check spelling."
        
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
            "and providing well-structured, factual responses. You understand how to "
            "interpret Wikipedia content and extract the most important insights for users."
        ),
        tools=[search_wikipedia],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_research_task(topic="artificial intelligence"):
    return Task(
        description=(
            f"Research '{topic}' using Wikipedia and provide a comprehensive overview. "
            f"Include key concepts, historical background, current developments, and "
            f"important details. Organize the information in a clear, structured format "
            f"that would be helpful for someone learning about this topic."
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

def check_dependencies():
    missing_deps = []
    
    try:
        import wikipedia
        wikipedia.search("test", results=1)
    except ImportError:
        missing_deps.append("wikipedia")
    except Exception:
        pass
    
    return missing_deps

def main():
    if not GEMINI_API_KEY:
        print("⚠️  Please set your GEMINI_API_KEY environment variable")
        return
    

    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️  Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nTo install missing dependencies, run:")
        print("pip install wikipedia")
        return
    
    print("🚀 Starting Wikipedia Research with Gemini 2.5 Flash...")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini LLM configured")
    
    # Create agent
    researcher = create_wikipedia_researcher(gemini_llm)
    print("✅ Wikipedia researcher agent created")
    
    # Create task 
    research_task = create_research_task("machine learning")
    research_task.agent = researcher
    print("✅ Research task configured")
    
    # Create and run crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )
    
    print("\n📚 Executing Wikipedia research...")
    print("=" * 50)
    
 
    result = crew.kickoff()
    print("\n" + "=" * 50)
    print("📖 WIKIPEDIA RESEARCH RESULTS")
    print("=" * 50)
    print(result)
        
    
def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
