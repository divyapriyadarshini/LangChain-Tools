import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper

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

@tool("Google Trends Search")
def search_google_trends(query: str) -> str:
    """
    Search Google Trends for trend data and popularity information.
    Args:
        query: Search term to analyze trends for (e.g., "artificial intelligence", "Tesla")
    Returns:
        String containing trend data with values, percentages, and related queries
    """
    try:
        if not SERP_API_KEY:  
            return "Error: SERP_API_KEY not found. Please set your SerpAPI key in environment variables."
        
        trends_wrapper = GoogleTrendsAPIWrapper(serp_api_key=SERP_API_KEY) 
        trends_tool = GoogleTrendsQueryRun(api_wrapper=trends_wrapper)
        results = trends_tool.run(query)
        return f"Google Trends data for '{query}':\n{results}"
    except Exception as e:
        return f"Error searching Google Trends: {str(e)}"

@tool("Google Trends Quick Analysis")
def quick_trends_analysis(query: str) -> str:
    """
    Get a quick trends analysis with summary insights.
    Args:
        query: Search term to analyze trends for
    Returns:
        Formatted summary of trend insights and analysis
    """
    try:
        if not SERP_API_KEY:  
            return "Error: SERP_API_KEY not found. Please set your SerpAPI key in environment variables."
        
        trends_wrapper = GoogleTrendsAPIWrapper(serp_api_key=SERP_API_KEY)  
        raw_data = trends_wrapper.run(query)
        
        lines = raw_data.split('\n')
        formatted_result = f"**Trends Analysis for '{query.title()}'**\n\n"
        
        for line in lines:
            if line.startswith('Query:'):
                formatted_result += f"**Search Term:** {line.split(':', 1)[1].strip()}\n"
            elif line.startswith('Date From:'):
                formatted_result += f"**Period:** {line.split(':', 1)[1].strip()}"
            elif line.startswith('Date To:'):
                formatted_result += f" to {line.split(':', 1)[1].strip()}\n"
            elif line.startswith('Min Value:'):
                formatted_result += f"**Minimum Interest:** {line.split(':', 1)[1].strip()}\n"
            elif line.startswith('Max Value:'):
                formatted_result += f"**Peak Interest:** {line.split(':', 1)[1].strip()}\n"
            elif line.startswith('Average Value:'):
                avg_val = float(line.split(':', 1)[1].strip())
                formatted_result += f"**Average Interest:** {avg_val:.1f}\n"
            elif line.startswith('Precent Change:'):
                formatted_result += f"**Trend Change:** {line.split(':', 1)[1].strip()}\n"
            elif line.startswith('Rising Related Queries:'):
                queries = line.split(':', 1)[1].strip()
                formatted_result += f"\n**Rising Topics:** {queries}\n"
            elif line.startswith('Top Related Queries:'):
                queries = line.split(':', 1)[1].strip()
                formatted_result += f"**Top Related Searches:** {queries}\n"
        
        return formatted_result
        
    except Exception as e:
        return f"Error in quick trends analysis: {str(e)}"

@tool("Google Trends Multi-Term Comparison")
def compare_trends(terms: str) -> str:
    """
    Compare trends for multiple search terms (comma-separated).
    Args:
        terms: Comma-separated search terms to compare (e.g., "iPhone,Samsung,Google Pixel")
    Returns:
        Comparison of trend data for multiple terms
    """
    try:
        if not SERP_API_KEY:  
            return "Error: SERP_API_KEY not found. Please set your SerpAPI key in environment variables."
        
        term_list = [term.strip() for term in terms.split(',')]
        if len(term_list) < 2:
            return "Please provide at least 2 terms separated by commas for comparison."
        
        trends_wrapper = GoogleTrendsAPIWrapper(serp_api_key=SERP_API_KEY) 
        comparison_results = f"**Trend Comparison Analysis**\n\n"
        
        for i, term in enumerate(term_list, 1):
            try:
                result = trends_wrapper.run(term)
                lines = result.split('\n')
                
                comparison_results += f"**{i}. {term.title()}**\n"
                for line in lines:
                    if line.startswith('Average Value:'):
                        avg_val = float(line.split(':', 1)[1].strip())
                        comparison_results += f"   Average Interest: {avg_val:.1f}\n"
                    elif line.startswith('Precent Change:'):
                        comparison_results += f"   Trend Change: {line.split(':', 1)[1].strip()}\n"
                    elif line.startswith('Max Value:'):
                        comparison_results += f"   Peak Interest: {line.split(':', 1)[1].strip()}\n"
                comparison_results += "\n"
            except Exception as e:
                comparison_results += f"   Error analyzing '{term}': {str(e)}\n\n"
        
        return comparison_results
        
    except Exception as e:
        return f"Error in trends comparison: {str(e)}"

def create_trends_agent(llm, tools):
    return Agent(
        role="Trends Analysis Specialist",
        goal="Analyze Google Trends data to identify search patterns, popularity trends, and market insights for various topics and keywords",
        backstory="You are an expert data analyst specializing in trend analysis and market research, with access to Google Trends data to provide insights on search behavior and topic popularity.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_trends_task(analysis_request, analysis_params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in analysis_params.items()]) if analysis_params else "None"
    
    task_description = f"""
    You have access to Google Trends analysis tools for understanding search behavior and topic popularity.

    ### Analysis Request:
    {analysis_request}

    ### Analysis Parameters:
    {params_str}

    Please:
    1. Analyze the request to determine the best trends analysis approach
    2. Use appropriate tools based on the request type:
    - Use "Google Trends Search" for detailed raw trend data
    - Use "Google Trends Quick Analysis" for summarized insights and formatted results
    - Use "Google Trends Multi-Term Comparison" for comparing multiple topics
    3. If needed, perform multiple searches to get comprehensive trend insights
    4. Provide clear interpretations of the trend data, including:
    - Popularity patterns and changes over time
    - Rising and declining interest
    - Related trending topics
    - Market insights and implications
    5. Format results in a clear, business-friendly manner

    Available tools:
    - Google Trends Search: Raw trend data with complete statistics
    - Google Trends Quick Analysis: Formatted summary with key insights
    - Google Trends Multi-Term Comparison: Side-by-side comparison of multiple terms
    """
    
    return Task(
        description=task_description,
        expected_output="Comprehensive trends analysis with insights, patterns, and actionable market intelligence",
        agent=None  # to be assigned later
    )


def get_user_input():
    print("Google Trends Analysis Tool")
    print("=" * 30)
    print("Analysis types:")
    print("1. Single topic trends analysis")
    print("2. Quick trends summary")
    print("3. Multi-term comparison")
    print("4. Market research analysis")
    
    choice = input("\nSelect analysis type (1-4): ").strip()
    analysis_params = {}
    
    if choice == "1":
        topic = input("Enter topic to analyze: ").strip()
        analysis_request = f"Perform detailed trends analysis for: {topic}"
        analysis_params = {"topic": topic, "analysis_type": "detailed"}
    elif choice == "2":
        topic = input("Enter topic for quick analysis: ").strip()
        analysis_request = f"Provide quick trends summary for: {topic}"
        analysis_params = {"topic": topic, "analysis_type": "quick"}
    elif choice == "3":
        terms = input("Enter terms to compare (comma-separated): ").strip()
        analysis_request = f"Compare trends for: {terms}"
        analysis_params = {"terms": terms, "analysis_type": "comparison"}
    elif choice == "4":
        industry = input("Enter industry/market to research: ").strip()
        analysis_request = f"Conduct market research trends analysis for: {industry}"
        analysis_params = {"industry": industry, "analysis_type": "market_research"}
    else:
        topic = input("Enter topic to analyze: ").strip() or "artificial intelligence"
        analysis_request = f"Analyze trends for: {topic}"
        analysis_params = {"topic": topic, "analysis_type": "general"}
    
    return analysis_request, analysis_params

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    if not SERP_API_KEY:  
        print("⚠️ Please set your SERP_API_KEY environment variable")
        print("   1. Go to https://serpapi.com/users/sign_up")
        return
    
    analysis_request, analysis_params = get_user_input()
    
    print(f"\n🚀 Starting Google Trends Analysis")
    print(f"Request: {analysis_request}")
    
    tools = [
        search_google_trends,
        quick_trends_analysis,
        compare_trends
    ]
    
    llm = setup_gemini_llm()
    agent = create_trends_agent(llm, tools)
    task = create_trends_task(analysis_request, analysis_params)
    task.agent = agent
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("\n📈 Conducting Google Trends analysis...")
    print("=" * 50)
    result = crew.kickoff()
    print("\n📊 Trends Analysis Results:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
