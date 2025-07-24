#!/usr/bin/env python3
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
    Solve mathematical problems using Wolfram Alpha.
    
    Args:
        query: Mathematical expression, scientific question, or factual query
    Returns:
        String containing the computed result or answer from Wolfram Alpha
    """
    try:
        wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_ALPHA_APPID)
        result = wolfram.run(query)
        
        if not result or result.strip() == "":
            return f"Wolfram Alpha could not compute an answer for: '{query}'. Please try rephrasing your question."
        
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
            "and providing accurate factual information across STEM fields."
        ),
        tools=[wolfram_alpha_query],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_computation_task(query):
    return Task(
        description=(
            f"Use Wolfram Alpha to solve this query: '{query}'. "
            f"Provide a comprehensive explanation of the result, including any "
            f"relevant mathematical concepts, steps involved, or contextual information."
        ),
        expected_output=(
            f"A detailed solution for '{query}' including:\n"
            "- The computed result from Wolfram Alpha\n"
            "- Explanation of the mathematical/scientific concepts involved\n"
            "- Step-by-step breakdown if applicable\n"
            "- Any relevant additional insights"
        ),
        agent=None
    )

def check_requirements():
    # Check API keys
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not WOLFRAM_ALPHA_APPID:
        missing.append("WOLFRAM_ALPHA_APPID")
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   - {var}")
        if "WOLFRAM_ALPHA_APPID" in missing:
            print("Get your Wolfram Alpha APP ID from:")
            print("https://developer.wolframalpha.com/portal/myapps")
        print("Create a .env file with these variables.")
        return False
    
    # Check dependencies
    missing_deps = []
    try:
        import wolframalpha
    except ImportError:
        missing_deps.append("wolframalpha")
    
    try:
        from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
    except ImportError:
        missing_deps.append("langchain-community")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("Install with: pip install wolframalpha langchain-community")
        return False
    
    return True

def main():
    if not check_requirements():
        return
    
    print("🧮 Wolfram Alpha Mathematical Calculator")
    print("=" * 40)
    
    try:
        # Setup AI
        llm = setup_gemini_llm()
        agent = create_computational_expert(llm)
        
        while True:
            # Get user input
            math_question = input("\n🔢 Enter your math question (or 'exit' to quit): ").strip()
            
            if math_question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not math_question:
                print("❌ Please enter a math question.")
                continue
            
            print(f"\n🚀 Solving: '{math_question}'...")
            print("-" * 40)
            
            # Create and run task
            task = create_computation_task(math_question)
            task.agent = agent
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False
            )
            
            result = crew.kickoff()
            print("\n" + "=" * 40)
            print("📊 SOLUTION:")
            print("=" * 40)
            print(result)
            print("=" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
