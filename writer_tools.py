#!/usr/bin/env python3
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
    Search the Writer Knowledge Graph for information.
    
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
            return "Error: No graph ID provided. Set WRITER_GRAPH_ID environment variable."
        
        graph_tool = GraphTool(graph_ids=[graph_id])
        return graph_tool.invoke(query)
        
    except Exception as e:
        return f"Error searching knowledge graph: {str(e)}"

@tool("Writer Chat Completion")
def writer_chat_completion(prompt: str, model: str = "palmyra-x5") -> str:
    """
    Generate text using Writer's chat models.
    
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
    Use Writer's no-code application.
    
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
            return "Error: No app ID provided. Set WRITER_APP_ID environment variable."
        
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

def create_writer_task(task_type, query):
    task_descriptions = {
        "knowledge": f"Use the Writer Knowledge Graph to find comprehensive information about '{query}'. Provide detailed insights and analysis.",
        "generate": f"Use Writer's chat models to generate high-quality content about '{query}'. Focus on accuracy and engagement.",
        "nocode": f"Use Writer's no-code application to process '{query}' and provide specialized results.",
        "all": f"Research '{query}' using all available Writer tools - knowledge graph, chat models, and no-code apps. Provide a comprehensive analysis."
    }
    
    expected_outputs = {
        "knowledge": "Detailed information from Writer's Knowledge Graph with key insights and references",
        "generate": "High-quality generated content using Writer's advanced language models",
        "nocode": "Specialized results from Writer's no-code application processing",
        "all": "Comprehensive analysis using multiple Writer tools with comparative insights"
    }
    
    return Task(
        description=task_descriptions.get(task_type, task_descriptions["generate"]),
        expected_output=expected_outputs.get(task_type, expected_outputs["generate"]),
        agent=None
    )

def get_task_type():
    print("\nChoose task type:")
    print("1. 📚 Knowledge Graph Search")
    print("2. ✍️  Text Generation") 
    print("3. 🛠️  NoCode App")
    print("4. 🔍 All Tools")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-4): "))
            if choice == 1:
                return "knowledge"
            elif choice == 2:
                return "generate"
            elif choice == 3:
                return "nocode"
            elif choice == 4:
                return "all"
            else:
                print("❌ Please enter 1, 2, 3, or 4")
        except ValueError:
            print("❌ Please enter a valid number")

def check_requirements():
    # Check API keys
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not WRITER_API_KEY:
        missing.append("WRITER_API_KEY")
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   - {var}")
        if "WRITER_API_KEY" in missing:
            print("Get your Writer API key from: https://writer.com/")
        print("Create a .env file with these variables.")
        return False
    
    # Check dependencies
    try:
        import langchain_writer
    except ImportError:
        print("❌ Missing dependency: langchain-writer")
        print("Install with: pip install langchain-writer")
        return False
    
    return True

def main():
    if not check_requirements():
        return
    
    print("🚀 Writer AI Tools")
    print("=" * 30)
    
    try:
        # Setup AI
        llm = setup_gemini_llm()
        agent = create_writer_specialist(llm)
        
        while True:
            # Get user input
            prompt = input("\n✍️  Enter your prompt (or 'exit' to quit): ").strip()
            
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not prompt:
                print("❌ Please enter a prompt.")
                continue
            
            # Get task type
            task_type = get_task_type()
            
            print(f"\n🚀 Processing with Writer AI: '{prompt}'...")
            print("-" * 30)
            
            # Create and run task
            task = create_writer_task(task_type, prompt)
            task.agent = agent
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False
            )
            
            result = crew.kickoff()
            print("\n" + "=" * 30)
            print("📄 RESULT:")
            print("=" * 30)
            print(result)
            print("=" * 30)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
