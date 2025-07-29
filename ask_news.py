import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools.asknews import AskNewsSearch

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("AskNews Current Search")
def search_current_news(query: str) -> str:
    """
    Search for current news and events using AskNews API.
    Args:
        query: Search query for news (e.g., "tech sector federal policy", "AI developments")
    Returns:
        String containing enriched news articles with summaries, entities, and classifications
    """
    try:
        if not ASKNEWS_CLIENT_ID or not ASKNEWS_CLIENT_SECRET:
            return "Error: ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET must be set in environment variables"
        
        tool = AskNewsSearch(max_results=5)
        results = tool.invoke({"query": query})
        return f"Current news for '{query}':\n{results}"
    except Exception as e:
        return f"Error searching current news: {str(e)}"

@tool("AskNews Historical Search")
def search_historical_news(query: str, hours_back: int = 168) -> str:
    """
    Search for historical news going back a specified number of hours.
    Args:
        query: Search query for news
        hours_back: Number of hours to search back (default 168 = 1 week)
    Returns:
        Historical news articles with enrichment data
    """
    try:
        if not ASKNEWS_CLIENT_ID or not ASKNEWS_CLIENT_SECRET:
            return "Error: ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET must be set in environment variables"
        
        tool = AskNewsSearch(max_results=7)
        enhanced_query = f"{query} from {hours_back} hours ago"
        results = tool.invoke({"query": enhanced_query})
        return f"Historical news ({hours_back}h back) for '{query}':\n{results}"
    except Exception as e:
        return f"Error searching historical news: {str(e)}"

@tool("AskNews Quick Brief")
def quick_news_brief(topic: str) -> str:
    """
    Get a quick news brief on a specific topic with top headlines.
    Args:
        topic: Topic to get news brief for (e.g., "artificial intelligence", "electric vehicles")
    Returns:
        Concise news brief with key highlights
    """
    try:
        if not ASKNEWS_CLIENT_ID or not ASKNEWS_CLIENT_SECRET:
            return "Error: ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET must be set in environment variables"
        
        tool = AskNewsSearch(max_results=3)
        results = tool.invoke({"query": f"latest {topic} news brief"})
        
        lines = results.split('\n')
        brief = f"**News Brief: {topic.title()}**\n\n"
        
        doc_count = 0
        for line in lines:
            if line.startswith('<doc>'):
                doc_count += 1
                brief += f"**Story {doc_count}:**\n"
            elif line.startswith('title:'):
                title = line.split(':', 1)[1].strip()
                brief += f"📰 {title}\n"
            elif line.startswith('summary:'):
                summary = line.split(':', 1)[1].strip()
                brief += f"   {summary}\n"
            elif line.startswith('source:'):
                source = line.split(':', 1)[1].strip()
                brief += f"   📍 Source: {source}\n\n"
        
        return brief
        
    except Exception as e:
        return f"Error creating news brief: {str(e)}"

def create_news_agent(llm, tools):
    return Agent(
        role="News Research Specialist",
        goal="Provide comprehensive, up-to-date news analysis and summaries using AskNews to deliver enriched, contextual information on current events and trends",
        backstory="You are an expert news analyst with access to AskNews API, capable of retrieving and analyzing over 300k daily news articles across 13 languages from 50k+ sources worldwide, providing enriched summaries with entities, classifications, and sentiment analysis.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_news_task(news_request, news_params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in news_params.items()]) if news_params else "None"
    
    task_description = f"""
You have access to AskNews tools for comprehensive news research and analysis.

### News Request:
{news_request}

### Search Parameters:
{params_str}

Please:
1. Analyze the news request to determine the best search approach
2. Use appropriate AskNews tools based on the request type:
   - Use "AskNews Current Search" for latest news and current events
   - Use "AskNews Historical Search" for news from specific time periods
   - Use "AskNews Quick Brief" for concise topic summaries
3. If needed, perform multiple searches to get comprehensive coverage
4. Provide analysis that includes:
   - Key developments and trends
   - Source diversity and credibility
   - Geographic and temporal context
   - Entity recognition and classifications
   - Sentiment analysis where relevant
5. Format results clearly with proper sourcing and timestamps

Available tools:
- AskNews Current Search: Latest news with enriched metadata and entity extraction
- AskNews Historical Search: Time-specific news analysis going back hours/days
- AskNews Quick Brief: Concise topic summaries with key highlights
"""
    
    return Task(
        description=task_description,
        expected_output="Comprehensive news analysis with enriched context, entity recognition, source attribution, and actionable insights",
        agent=None  # to be assigned later
    )

def get_user_input():
    print("AskNews Research Tool")
    print("=" * 25)
    print("News research types:")
    print("1. Current news search")
    print("2. Historical news search")
    print("3. Quick news brief")
    print("4. Comprehensive news analysis")
    
    choice = input("\nSelect research type (1-4): ").strip()
    news_params = {}
    
    if choice == "1":
        topic = input("Enter news topic: ").strip()
        news_request = f"Search for current news about: {topic}"
        news_params = {"topic": topic, "search_type": "current"}
    elif choice == "2":
        topic = input("Enter news topic: ").strip()
        try:
            hours = int(input("Hours to search back (default 168 = 1 week): ").strip() or "168")
        except ValueError:
            hours = 168
        news_request = f"Search for historical news about: {topic}"
        news_params = {"topic": topic, "hours_back": hours, "search_type": "historical"}
    elif choice == "3":
        topic = input("Enter topic for news brief: ").strip()
        news_request = f"Create a quick news brief for: {topic}"
        news_params = {"topic": topic, "search_type": "brief"}
    elif choice == "4":
        topic = input("Enter topic for comprehensive analysis: ").strip()
        news_request = f"Conduct comprehensive news analysis on: {topic}"
        news_params = {"topic": topic, "search_type": "comprehensive"}
    else:
        topic = input("Enter news topic: ").strip() or "artificial intelligence"
        news_request = f"Search for news about: {topic}"
        news_params = {"topic": topic, "search_type": "general"}
    
    return news_request, news_params

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_CLIENT_SECRET:
        print("⚠️ Please set your AskNews API credentials:")
        print(" Go to https://asknews.app")
        return
    
    news_request, news_params = get_user_input()
    
    print(f"\n Starting AskNews Research")
    print(f"Request: {news_request}")
    
    tools = [
        search_current_news,
        search_historical_news,
        quick_news_brief
    ]
    
    llm = setup_gemini_llm()
    agent = create_news_agent(llm, tools)
    task = create_news_task(news_request, news_params)
    task.agent = agent
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("\n Conducting AskNews research...")
    print("=" * 50)
    result = crew.kickoff()
    print("\n News Analysis Results:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
