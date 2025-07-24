import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Yahoo Finance News Search")
def search_finance_news(ticker: str) -> str:
    """
    Search for financial news about a public company using ticker symbol.
    Args:
        ticker: Company ticker symbol (e.g., AAPL for Apple, MSFT for Microsoft)
    Returns:
        String containing recent financial news about the company
    """
    try:   
        finance_tool = YahooFinanceNewsTool()
        results = finance_tool.invoke(ticker.upper())
        
        if "No news found" in results:
            return f"No financial news found for ticker symbol '{ticker.upper()}'. Please verify the ticker symbol is correct."
        
        return results

    except Exception as e:
        return f"Error searching Yahoo Finance news: {str(e)}"

def create_finance_analyst(llm):
    return Agent(
        role="Financial News Analyst",
        goal="Search and analyze the latest financial news for public companies using ticker symbols",
        backstory=(
            "You are an expert financial analyst who specializes in tracking "
            "market news and developments for publicly traded companies. You "
            "have deep knowledge of stock tickers and can quickly identify "
            "relevant financial news that impacts company performance and "
            "market sentiment."
        ),
        tools=[search_finance_news],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_analysis_task(ticker):
    return Task(
        description=(
            f"Search for the latest financial news about {ticker.upper()} and provide "
            f"a comprehensive analysis. Look for recent developments, earnings news, "
            f"market trends, and any significant events that could impact the company's "
            f"stock performance. Summarize the key findings and their potential implications."
        ),
        expected_output=(
            f"A comprehensive financial news analysis for {ticker.upper()} including:\n"
            "- Summary of recent news headlines\n"
            "- Key financial developments or announcements\n"
            "- Market sentiment and trends\n"
            "- Potential impact on stock performance\n"
            "- Overall assessment of current financial position"
        ),
        agent=None
    )

def get_user_input():
    """Get ticker symbol from user"""
    print("Yahoo Finance News Analysis Tool")
    print("=" * 40)
    
    # Get ticker symbol
    ticker = input("Enter ticker symbol to analyze: ").strip()
    if not ticker:
        print("⚠️ No ticker entered. Using default: AAPL")
        ticker = "AAPL"
    
    return ticker.upper()

def check_dependencies():
    missing_deps = []
    
    try:
        import yfinance
    except ImportError:
        missing_deps.append("yfinance")
    
    try:
        from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
    except ImportError:
        missing_deps.append("langchain-community")
    
    return missing_deps

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return

    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nTo install missing dependencies, run:")
        if "yfinance" in missing_deps:
            print("pip install yfinance")
        if "langchain-community" in missing_deps:
            print("pip install langchain-community")
        return
    
    # Get user input
    ticker = get_user_input()
    
    print(f"\n🚀 Starting Yahoo Finance News Analysis for {ticker}")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini 2.5 Flash LLM configured")
    
    # Create agent
    analyst = create_finance_analyst(gemini_llm)
    print("✅ Financial news analyst agent created")
    
    # Create task with user input
    analysis_task = create_analysis_task(ticker)
    analysis_task.agent = analyst
    print("✅ Financial analysis task configured")
    
    # Create and run crew
    crew = Crew(
        agents=[analyst],
        tasks=[analysis_task],
        verbose=True
    )
    
    print(f"\n💰 Executing Yahoo Finance news search for {ticker}...")
    print("=" * 50)

    result = crew.kickoff()
    print(result)

def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()

