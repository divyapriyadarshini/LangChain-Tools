import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Wikidata Knowledge Query")
def query_wikidata(entity: str) -> str:
    """
    Query Wikidata for information about an entity.
    
    Args:
        entity: Name of person, place, organization, or concept to search for
    Returns:
        String containing structured Wikidata information with properties and relationships
    """
    try:
        wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
        result = wikidata.run(entity)
        
        if not result or result.strip() == "":
            return f"No Wikidata information found for '{entity}'. Please try a different search term."
        
        return f"**Wikidata Information for '{entity}':**\n\n{result}"
        
    except Exception as e:
        return f"Error querying Wikidata: {str(e)}"

def create_wikidata_specialist(llm):
    return Agent(
        role="Wikidata Knowledge Specialist",
        goal="Query and analyze structured data from Wikidata to provide comprehensive factual information about entities",
        backstory=(
            "You are an expert in structured knowledge bases with access to Wikidata's "
            "vast repository of linked data. You excel at extracting detailed factual "
            "information about people, places, organizations, concepts, and their relationships."
        ),
        tools=[query_wikidata],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_knowledge_task(entity):
    return Task(
        description=(
            f"Query Wikidata for comprehensive information about '{entity}'. "
            f"Extract and analyze all relevant properties, relationships, and structured data. "
            f"Present the information in a clear, organized format."
        ),
        expected_output=(
            f"A comprehensive knowledge report on '{entity}' with key facts, "
            "relationships, and structured data from Wikidata"
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
    missing_deps = []
    try:
        import wikibase_rest_api_client
    except ImportError:
        missing_deps.append("wikibase-rest-api-client")
    
    try:
        import mediawikiapi
    except ImportError:
        missing_deps.append("mediawikiapi")
    
    try:
        from langchain_community.tools.wikidata.tool import WikidataAPIWrapper
    except ImportError:
        missing_deps.append("langchain-community")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with:")
        print("pip install wikibase-rest-api-client mediawikiapi langchain-community")
        return False
    
    return True

def main():
    if not check_requirements():
        return
    
    print("🚀 Wikidata Knowledge Explorer")
    print("=" * 40)
    
    try:
        # Setup AI
        llm = setup_gemini_llm()
        agent = create_wikidata_specialist(llm)
        
        while True:
            # Get user input
            topic = input("\n📚 Enter topic to search (or 'exit' to quit): ").strip()
            
            if topic.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not topic:
                print("❌ Please enter a topic.")
                continue
            
            print(f"\n🔍 Searching Wikidata for: '{topic}'...")
            print("-" * 40)
            
            # Create and run task
            task = create_knowledge_task(topic)
            task.agent = agent
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False
            )
            
            result = crew.kickoff()
            print("\n" + "=" * 40)
            print("📊 RESULT:")
            print("=" * 40)
            print(result)
            print("=" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
