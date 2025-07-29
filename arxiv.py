import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("ArXiv Paper Search")
def search_arxiv_papers(query: str) -> str:
    """
    Search for scientific papers on ArXiv using direct API calls.
    Args:
        query: Search query (e.g., "neural networks", "machine learning")
    Returns:
        String containing paper details with title, authors, summary, and publication date
    """
    try:
        # Direct ArXiv API call
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': 3,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return f"Error: ArXiv API returned status {response.status_code}"
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        if not entries:
            return f"No papers found for query: {query}"
        
        results = f"ArXiv search results for '{query}':\n\n"
        
        for i, entry in enumerate(entries, 1):
            # Extract paper details
            title = entry.find('atom:title', ns)
            title_text = title.text.strip().replace('\n', ' ') if title is not None else "No Title"
            
            # Authors
            authors = entry.findall('atom:author', ns)
            author_names = []
            for author in authors:
                name = author.find('atom:name', ns)
                if name is not None:
                    author_names.append(name.text)
            
            # Publication date
            published = entry.find('atom:published', ns)
            pub_date = published.text[:10] if published is not None else "Unknown"
            
            # Summary
            summary = entry.find('atom:summary', ns)
            summary_text = summary.text.strip()[:500] + "..." if summary is not None else "No summary"
            
            # ArXiv ID
            entry_id = entry.find('atom:id', ns)
            arxiv_id = entry_id.text.split('/')[-1] if entry_id is not None else "Unknown"
            
            # PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            results += f"**Paper {i}:**\n"
            results += f"📄 Title: {title_text}\n"
            results += f"👥 Authors: {', '.join(author_names)}\n"
            results += f"📅 Published: {pub_date}\n"
            results += f"🆔 ArXiv ID: {arxiv_id}\n"
            results += f"📝 Summary: {summary_text}\n"
            results += f"🔗 PDF: {pdf_url}\n\n"
        
        return results
        
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

@tool("ArXiv Paper Details")
def get_paper_details(arxiv_id: str) -> str:
    """
    Get detailed information about a specific ArXiv paper by ID.
    Args:
        arxiv_id: ArXiv paper ID (e.g., "1706.03762", "2010.11929")
    Returns:
        Detailed paper information including full summary and metadata
    """
    try:
        # Clean the ArXiv ID
        clean_id = arxiv_id.replace('v1', '').replace('v2', '').replace('v3', '')
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'id_list': clean_id,
            'start': 0,
            'max_results': 1
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return f"Error: ArXiv API returned status {response.status_code}"
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            return f"No paper found with ArXiv ID: {arxiv_id}"
        
        # Extract detailed information
        title = entry.find('atom:title', ns)
        title_text = title.text.strip().replace('\n', ' ') if title is not None else "No Title"
        
        authors = entry.findall('atom:author', ns)
        author_names = []
        for author in authors:
            name = author.find('atom:name', ns)
            if name is not None:
                author_names.append(name.text)
        
        published = entry.find('atom:published', ns)
        pub_date = published.text[:10] if published is not None else "Unknown"
        
        updated = entry.find('atom:updated', ns)
        update_date = updated.text[:10] if updated is not None else "Unknown"
        
        summary = entry.find('atom:summary', ns)
        summary_text = summary.text.strip() if summary is not None else "No summary"
        
        categories = entry.findall('atom:category', ns)
        cat_list = [cat.get('term') for cat in categories if cat.get('term')]
        
        result_text = f"""
**Paper Details for ArXiv ID: {arxiv_id}**

📄 **Title:** {title_text}
👥 **Authors:** {', '.join(author_names)}
📅 **Published:** {pub_date}
🔄 **Updated:** {update_date}
🏷️ **Categories:** {', '.join(cat_list)}
🔗 **PDF:** https://arxiv.org/pdf/{clean_id}.pdf

**Abstract:**
{summary_text}
"""
        return result_text
        
    except Exception as e:
        return f"Error retrieving paper details: {str(e)}"

@tool("ArXiv Topic Research")
def research_arxiv_topic(topic: str, max_papers: int = 5) -> str:
    """
    Conduct comprehensive research on a scientific topic using ArXiv.
    Args:
        topic: Research topic (e.g., "quantum computing", "neural networks")
        max_papers: Maximum number of papers to retrieve (1-10)
    Returns:
        Comprehensive research summary with multiple relevant papers
    """
    try:
        max_papers = min(max(max_papers, 1), 10)
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{topic}',
            'start': 0,
            'max_results': max_papers,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return f"Error: ArXiv API returned status {response.status_code}"
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        if not entries:
            return f"No papers found for topic: {topic}"
        
        formatted_results = f"**Research Summary: {topic.title()}**\n\n"
        
        for i, entry in enumerate(entries, 1):
            title = entry.find('atom:title', ns)
            title_text = title.text.strip().replace('\n', ' ') if title is not None else "No Title"
            
            authors = entry.findall('atom:author', ns)
            author_names = []
            for author in authors:
                name = author.find('atom:name', ns)
                if name is not None:
                    author_names.append(name.text)
            
            published = entry.find('atom:published', ns)
            pub_date = published.text[:10] if published is not None else "Unknown"
            
            summary = entry.find('atom:summary', ns)
            summary_text = summary.text.strip()[:400] + "..." if summary is not None else "No summary"
            
            entry_id = entry.find('atom:id', ns)
            arxiv_id = entry_id.text.split('/')[-1] if entry_id is not None else "Unknown"
            
            categories = entry.findall('atom:category', ns)
            cat_list = [cat.get('term') for cat in categories if cat.get('term')]
            
            formatted_results += f"**Paper {i}:**\n"
            formatted_results += f"📅 Published: {pub_date}\n"
            formatted_results += f"📄 Title: {title_text}\n"
            formatted_results += f"👥 Authors: {', '.join(author_names[:3])}{'...' if len(author_names) > 3 else ''}\n"
            formatted_results += f"🏷️ Categories: {', '.join(cat_list[:3])}\n"
            formatted_results += f"📝 Summary: {summary_text}\n"
            formatted_results += f"🆔 ArXiv ID: {arxiv_id}\n"
            formatted_results += f"🔗 PDF: https://arxiv.org/pdf/{arxiv_id}.pdf\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error conducting topic research: {str(e)}"

def create_arxiv_agent(llm, tools):
    return Agent(
        role="Scientific Research Specialist",
        goal="Conduct comprehensive scientific literature research using ArXiv to find relevant academic papers, analyze research trends, and provide detailed summaries across multiple disciplines",
        backstory="You are an expert scientific researcher with access to ArXiv's vast collection of over 2 million scholarly articles in physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering, and economics.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_arxiv_task(research_request, research_params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in research_params.items()]) if research_params else "None"
    
    task_description = f"""
You have access to ArXiv research tools for comprehensive scientific literature analysis.

### Research Request:
{research_request}

### Research Parameters:
{params_str}

Please:
1. Analyze the research request to determine the best search approach
2. Use appropriate ArXiv tools based on the request type:
   - Use "ArXiv Paper Search" for general topic searches and finding relevant papers
   - Use "ArXiv Paper Details" for detailed information about specific papers by ID
   - Use "ArXiv Topic Research" for comprehensive multi-paper analysis on specific topics
3. If needed, perform multiple searches to get comprehensive research coverage
4. Provide analysis that includes:
   - Paper summaries and key findings
   - Author information and publication dates
   - Research trends and methodologies
   - Connections between different papers
   - Implications for the field
5. Format results clearly with proper paper citations and academic context

Available tools:
- ArXiv Paper Search: General search across ArXiv's scientific paper database
- ArXiv Paper Details: Detailed information retrieval for specific ArXiv paper IDs
- ArXiv Topic Research: Comprehensive multi-paper research analysis with formatted summaries

Disciplines covered: Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, Economics
"""
    
    return Task(
        description=task_description,
        expected_output="Comprehensive scientific literature analysis with paper summaries, research insights, and academic context",
        agent=None  
    )

def get_user_input():
    print("ArXiv Scientific Research Tool")
    print("=" * 32)
    print("Research types:")
    print("1. Search papers by topic")
    print("2. Get specific paper details by ArXiv ID")
    print("3. Comprehensive topic research")
    print("4. Academic trend analysis")
    
    choice = input("\nSelect research type (1-4): ").strip()
    research_params = {}
    
    if choice == "1":
        topic = input("Enter research topic: ").strip()
        research_request = f"Search for papers related to: {topic}"
        research_params = {"topic": topic, "search_type": "topic_search"}
    elif choice == "2":
        arxiv_id = input("Enter ArXiv paper ID (e.g., 1706.03762): ").strip()
        research_request = f"Get detailed information for ArXiv paper: {arxiv_id}"
        research_params = {"arxiv_id": arxiv_id, "search_type": "paper_details"}
    elif choice == "3":
        topic = input("Enter topic for comprehensive research: ").strip()
        try:
            max_papers = int(input("Number of papers to analyze (1-10, default 5): ").strip() or "5")
            max_papers = min(max(max_papers, 1), 10)
        except ValueError:
            max_papers = 5
        research_request = f"Conduct comprehensive research on: {topic}"
        research_params = {"topic": topic, "max_papers": max_papers, "search_type": "comprehensive"}
    elif choice == "4":
        field = input("Enter scientific field for trend analysis: ").strip()
        research_request = f"Analyze recent research trends in: {field}"
        research_params = {"field": field, "search_type": "trend_analysis"}
    else:
        topic = input("Enter research topic: ").strip() or "machine learning"
        research_request = f"Research papers on: {topic}"
        research_params = {"topic": topic, "search_type": "general"}
    
    return research_request, research_params

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    print("ArXiv Research Tool - No additional API keys required.")
    
    research_request, research_params = get_user_input()
    
    print(f"\n🚀 Starting ArXiv Scientific Research")
    print(f"Request: {research_request}")
    
    tools = [
        search_arxiv_papers,
        get_paper_details,
        research_arxiv_topic
    ]
    
    llm = setup_gemini_llm()
    agent = create_arxiv_agent(llm, tools)
    task = create_arxiv_task(research_request, research_params)
    task.agent = agent
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("\n📖 Conducting ArXiv research...")
    print("=" * 50)
    result = crew.kickoff()
    print("\n📊 Research Results:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()
