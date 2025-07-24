import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
             

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WOLFRAM_ALPHA_APPID = os.getenv("WOLFRAM_ALPHA_APPID")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Wolfram Alpha Calculator")
def wolfram_alpha_query(query: str) -> str:
    """
    Args:
        query: Mathematical expression, scientific question, or factual query
    Returns:
        String containing the computed result or answer from Wolfram Alpha
    """
    try:
        wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_ALPHA_APPID)
        result = wolfram.run(query)
        
        if not result or result.strip() == "":
            return f"Wolfram Alpha could not compute an answer for: '{query}'. Please try rephrasing your question or check if it's a valid mathematical/scientific query."
        
        return result
        
    except Exception as e:
        return f"Error querying Wolfram Alpha: {str(e)}"

def create_computational_expert(llm):
    return Agent(
        role="Computational Mathematics Expert",
        goal="Solve mathematical problems, perform scientific calculations, and answer factual queries using Wolfram Alpha",
        backstory=(
            "You are a brilliant computational mathematician and scientist with access "
            "to Wolfram Alpha's vast computational knowledge. You excel at solving "
            "complex mathematical equations, performing scientific calculations, "
            "analyzing data, and providing accurate factual information across "
            "mathematics, physics, chemistry, engineering, and other STEM fields."
        ),
        tools=[wolfram_alpha_query],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_computation_task(query="What is the derivative of x^3 + 2x^2 - 5x + 1?"):
    return Task(
        description=(
            f"Use Wolfram Alpha to solve this query: '{query}'. "
            f"Provide a comprehensive explanation of the result, including any "
            f"relevant mathematical concepts, steps involved, or contextual information. "
            f"If the query involves multiple parts or interpretations, address them all."
        ),
        expected_output=(
            f"A detailed solution for '{query}' including:\n"
            "- The computed result from Wolfram Alpha\n"
            "- Explanation of the mathematical/scientific concepts involved\n"
            "- Step-by-step breakdown if applicable\n"
            "- Any relevant additional insights or related information"
        ),
        agent=None
    )

def check_dependencies():
    missing_deps = []
    
    try:
        import wolframalpha
    except ImportError:
        missing_deps.append("wolframalpha")
    
    try:
        from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
    except ImportError:
        missing_deps.append("langchain-community")
    
    return missing_deps

def main():
    if not GEMINI_API_KEY:
        print("⚠️  Please set your GEMINI_API_KEY environment variable")
        return
    
    if not WOLFRAM_ALPHA_APPID:
        print("⚠️  Please set your WOLFRAM_ALPHA_APPID environment variable")
        print("Get your APP ID from: https://developer.wolframalpha.com/portal/myapps")
        return
    
 
    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️  Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nTo install missing dependencies, run:")
        if "wolframalpha" in missing_deps:
            print("pip install wolframalpha")
        if "langchain-community" in missing_deps:
            print("pip install langchain-community")
        return
    
    print("🚀 Starting Wolfram Alpha Computation with Gemini 2.5 Flash...")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini 2.5 Flash LLM configured")
    
    # Create agent
    expert = create_computational_expert(gemini_llm)
    print("✅ Computational expert agent created")
    
    # Create task 
    computation_task = create_computation_task("solve 2*x + 5 = -3*x + 7")
    computation_task.agent = expert
    print("✅ Computation task configured")
    
    # Create and run crew
    crew = Crew(
        agents=[expert],
        tasks=[computation_task],
        verbose=True
    )
    
    print("\n🧮 Executing Wolfram Alpha computation...")
    print("=" * 50)
    

    result = crew.kickoff()
    print("=" * 50)
    print(result)
        
def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
