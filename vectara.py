import os
import json
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_vectara import Vectara
from langchain_vectara.tools import VectaraRAG
        

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
VECTARA_CORPUS_KEY = os.getenv("VECTARA_CORPUS_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Vectara RAG Search")
def vectara_rag_search(query: str) -> str:
    """
    Args:
        query: Question or topic to research using your private corpus
    Returns:
        Generated answer with factual consistency score
    """
    try:
        vectara = Vectara(vectara_api_key=VECTARA_API_KEY)
        
        vectara_rag_tool = VectaraRAG(
            name="rag-tool",
            description="Get answers using RAG",
            vectorstore=vectara,
            corpus_key=VECTARA_CORPUS_KEY,
        )
        
        result = vectara_rag_tool.run(query)
        
        try:
            result_dict = json.loads(result)
            summary = result_dict.get("summary", result)
            fcs = result_dict.get("factual_consistency_score", "N/A")
            
            formatted_result = f"**Answer:** {summary}\n\n"
            if fcs != "N/A":
                formatted_result += f"**Factual Consistency Score:** {fcs}\n"
                formatted_result += "(Higher scores indicate higher confidence in factual accuracy)"
            
            return formatted_result
        except (json.JSONDecodeError, TypeError):
            return f"**Answer:** {result}"
        
    except ImportError:
        return "Error: Install dependencies with: pip install langchain-vectara"
    except Exception as e:
        return f"Error with Vectara RAG: {str(e)}"

def create_vectara_researcher(llm):
    return Agent(
        role="Vectara RAG Specialist",
        goal="Use Vectara's RAG capabilities to provide comprehensive, accurate answers from your private corpus",
        backstory=(
            "You are an expert at using Vectara's Retrieval Augmented Generation "
            "to find and synthesize information from private document collections. "
            "You excel at generating factually consistent answers backed by "
            "relevant source material."
        ),
        tools=[vectara_rag_search],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_rag_task(query="What are the key benefits of using AI?"):
    return Task(
        description=(
            f"Use Vectara RAG to research and answer: '{query}'. "
            f"Provide a comprehensive response based on the information "
            f"available in the corpus, including any relevant details "
            f"and factual consistency indicators."
        ),
        expected_output=(
            f"A comprehensive answer to '{query}' including:\n"
            "- Main response based on corpus information\n"
            "- Factual consistency score if available\n"
            "- Clear, well-structured information"
        ),
        agent=None
    )

def check_dependencies():
    missing_deps = []
    
    try:
        import langchain_vectara
    except ImportError:
        missing_deps.append("langchain-vectara")
    
    return missing_deps

def main():
    if not GEMINI_API_KEY:
        print("⚠️  Please set your GEMINI_API_KEY environment variable")
        return
    
    if not VECTARA_API_KEY:
        print("⚠️  Please set your VECTARA_API_KEY environment variable")
        print("Get your API key from: https://vectara.com/")
        return
    
    if not VECTARA_CORPUS_KEY:
        print("⚠️  Please set your VECTARA_CORPUS_KEY environment variable")
        print("This identifies your specific corpus in Vectara")
        return
    
    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️  Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nTo install missing dependencies, run:")
        print("pip install langchain-vectara")
        return
    
    print("🚀 Starting Vectara RAG Test with Gemini 2.5 Flash...")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini 2.5 Flash LLM configured")
    
    # Create agent
    researcher = create_vectara_researcher(gemini_llm)
    print("✅ Vectara RAG specialist agent created")
    
    # Create task 
    rag_task = create_rag_task("What is artificial intelligence?")
    rag_task.agent = researcher
    print("✅ RAG task configured")
    
    # Create and run crew
    crew = Crew(
        agents=[researcher],
        tasks=[rag_task],
        verbose=True
    )
    
    print("\n🔍 Executing Vectara RAG search...")
    print("=" * 50)
    
    # Execute the crew
    result = crew.kickoff()
    print("\n" + "=" * 50)
    print("🎯 VECTARA RAG RESULTS")
    print(result)
        
def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
