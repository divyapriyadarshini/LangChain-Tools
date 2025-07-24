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
    Args:
        entity: Name of person, place, organization, or concept to search for
    Returns:
        String containing structured Wikidata information with properties and relationships
    """
    try:
        wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
        result = wikidata.run(entity)
        
        if not result or result.strip() == "":
            return f"No Wikidata information found for '{entity}'. Please try a different search term or check spelling."
        
        if "Result Q" in result:
            formatted_result = "**Wikidata Knowledge Base Results:**\n\n"
            
            results = result.split("Result Q")
            for i, res in enumerate(results):
                if res.strip():
                    if i == 0:
                        continue 
                    formatted_result += f"**Entity Q{res.split(':')[0]}:**\n"
                    formatted_result += f"Q{res}\n\n"
            
            return formatted_result
        else:
            return f"**Wikidata Information for '{entity}':**\n\n{result}"
        
    except Exception as e:
        return f"Error querying Wikidata: {str(e)}"

def create_wikidata_specialist(llm):
    """Create Wikidata knowledge specialist agent"""
    return Agent(
        role="Wikidata Knowledge Specialist",
        goal="Query and analyze structured data from Wikidata to provide comprehensive factual information about entities",
        backstory=(
            "You are an expert in structured knowledge bases with access to Wikidata's "
            "vast repository of linked data. You excel at extracting detailed factual "
            "information about people, places, organizations, concepts, and their "
            "relationships. You understand how to interpret Wikidata's structured "
            "format and present complex entity relationships in a clear, organized manner."
        ),
        tools=[query_wikidata],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_knowledge_task(entity="Alan Turing"):
    """Create Wikidata knowledge extraction task"""
    return Task(
        description=(
            f"Query Wikidata for comprehensive information about '{entity}'. "
            f"Extract and analyze all relevant properties, relationships, and "
            f"structured data. Organize the information to highlight key facts, "
            f"biographical details, professional information, and notable achievements. "
            f"Present the data in a clear, structured format that showcases the "
            f"wealth of information available in Wikidata's knowledge base."
        ),
        expected_output=(
            f"A comprehensive knowledge report on '{entity}' including:\n"
            "- Entity identification and basic description\n"
            "- Key biographical or foundational information\n"
            "- Professional details and accomplishments\n"
            "- Important relationships and associations\n"
            "- Notable works, contributions, or characteristics\n"
            "- Structured data properties from Wikidata"
        ),
        agent=None
    )

def check_dependencies():
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
        from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
    except ImportError:
        missing_deps.append("langchain-community")
    
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
        print("pip install wikibase-rest-api-client mediawikiapi langchain-community")
        return
    
    print("🚀 Starting Wikidata Knowledge Query with Gemini 2.5 Flash...")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini 2.5 Flash LLM configured")
    
    # Create agent
    specialist = create_wikidata_specialist(gemini_llm)
    print("✅ Wikidata knowledge specialist agent created")
    
    # Create task - you can change the entity here
    knowledge_task = create_knowledge_task("Alan Turing")
    knowledge_task.agent = specialist
    print("✅ Knowledge extraction task configured")
    
    # Create and run crew
    crew = Crew(
        agents=[specialist],
        tasks=[knowledge_task],
        verbose=True
    )
    
    print("\n🔍 Executing Wikidata knowledge query...")
    print("=" * 50)
    

    result = crew.kickoff()
    print("\n" + "=" * 50)
    print("📊 WIKIDATA KNOWLEDGE RESULTS")
    print(result)
        

def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
