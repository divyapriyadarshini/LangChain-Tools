import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_writer.tools import GraphTool
from langchain_writer import ChatWriter
from langchain_writer.tools import NoCodeAppTool
        


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WRITER_API_KEY = os.getenv("WRITER_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Writer Knowledge Graph Search")
def search_knowledge_graph(query: str, graph_id: str = None) -> str:
    """
    Args:
        query: Search query for the knowledge graph
        graph_id: Optional graph ID (will use environment variable if not provided)
    Returns:
        String containing knowledge graph search results
    """
    try:        
        if not graph_id:
            graph_id = os.getenv("WRITER_GRAPH_ID")
            
        if not graph_id:
            return "Error: No graph ID provided. Set WRITER_GRAPH_ID environment variable or pass graph_id parameter."
        
        graph_tool = GraphTool(graph_ids=[graph_id])
        return graph_tool.invoke(query)
        
    except Exception as e:
        return f"Error searching knowledge graph: {str(e)}"

@tool("Writer Chat Completion")
def writer_chat_completion(prompt: str, model: str = "palmyra-x5") -> str:
    """
    Args:
        prompt: Text prompt for generation
        model: Writer model to use (default: palmyra-x5)
    Returns:
        Generated text response
    """
    try:
        chat = ChatWriter(
            model=model,
            temperature=0.7,
            api_key=WRITER_API_KEY
        )
        
        response = chat.invoke(prompt)
        return response.content
        
    except ImportError:
        return "Error: Install dependencies with: pip install langchain-writer"
    except Exception as e:
        return f"Error with Writer chat completion: {str(e)}"

@tool("Writer NoCode App")
def use_nocode_app(query: str, app_id: str = None) -> str:
    """
    Args:
        query: Input for the no-code application
        app_id: No-code app ID (will use environment variable if not provided)
    Returns:
        Result from the no-code application
    """
    try:
        if not app_id:
            app_id = os.getenv("WRITER_APP_ID")
            
        if not app_id:
            return "Error: No app ID provided. Set WRITER_APP_ID environment variable or pass app_id parameter."
        
        app_tool = NoCodeAppTool(
            app_id=app_id,
            name="Writer NoCode Application",
            description="No-code application for specialized tasks"
        )
        
        result = app_tool.run(tool_input={"inputs": {"query": query}})
        return str(result)
        
    except ImportError:
        return "Error: Install dependencies with: pip install langchain-writer"
    except Exception as e:
        return f"Error using no-code app: {str(e)}"

def create_writer_specialist(llm):
    return Agent(
        role="Writer AI Specialist",
        goal="Leverage Writer's AI capabilities including knowledge graphs, chat models, and no-code applications",
        backstory=(
            "You are an expert in using Writer's AI platform and tools. You have "
            "access to Writer's knowledge graphs for factual information retrieval, "
            "advanced chat models for text generation, and no-code applications for "
            "specialized tasks. You know how to choose the right tool for each situation."
        ),
        tools=[search_knowledge_graph, writer_chat_completion, use_nocode_app],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_writer_task(task_type="knowledge_search", query="artificial intelligence"):
    task_descriptions = {
        "knowledge_search": f"Use the Writer Knowledge Graph to find comprehensive information about '{query}'. Provide detailed insights and analysis.",
        "text_generation": f"Use Writer's chat models to generate high-quality content about '{query}'. Focus on accuracy and engagement.",
        "nocode_app": f"Use Writer's no-code application to process '{query}' and provide specialized results.",
        "comprehensive": f"Research '{query}' using all available Writer tools - knowledge graph, chat models, and no-code apps. Provide a comprehensive analysis."
    }
    
    expected_outputs = {
        "knowledge_search": "Detailed information from Writer's Knowledge Graph with key insights and references",
        "text_generation": "High-quality generated content using Writer's advanced language models",
        "nocode_app": "Specialized results from Writer's no-code application processing",
        "comprehensive": "Comprehensive analysis using multiple Writer tools with comparative insights"
    }
    
    return Task(
        description=task_descriptions.get(task_type, task_descriptions["comprehensive"]),
        expected_output=expected_outputs.get(task_type, expected_outputs["comprehensive"]),
        agent=None
    )

def check_dependencies():
    missing_deps = []
    
    try:
        import langchain_writer
    except ImportError:
        missing_deps.append("langchain-writer")
    
    return missing_deps

def main():
    if not GEMINI_API_KEY:
        print("⚠️  Please set your GEMINI_API_KEY environment variable")
        return
    
    if not WRITER_API_KEY:
        print("⚠️  Please set your WRITER_API_KEY environment variable")
        print("Get your API key from: https://writer.com/")
        return
    
    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️  Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nTo install missing dependencies, run:")
        print("pip install langchain-writer")
        return
    
    print("🚀 Starting Writer AI Tools with Gemini 2.5 Flash...")
    
    # Setup Gemini LLM
    gemini_llm = setup_gemini_llm()
    print("✅ Gemini 2.5 Flash LLM configured")
    
    # Create agent
    specialist = create_writer_specialist(gemini_llm)
    print("✅ Writer AI specialist agent created")
    
    # Create task - you can change the task type and query here
    writer_task = create_writer_task("text_generation", "future of artificial intelligence")
    writer_task.agent = specialist
    print("✅ Writer task configured")
    
    # Create and run crew
    crew = Crew(
        agents=[specialist],
        tasks=[writer_task],
        verbose=True
    )
    
    print("\n✍️  Executing Writer AI tools...")
    print("=" * 50)
    

    result = crew.kickoff()
    print(result)

def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
