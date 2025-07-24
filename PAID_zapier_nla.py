import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.utilities.zapier import ZapierNLAWrapper


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ZAPIER_NLA_API_KEY = os.getenv("ZAPIER_NLA_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096,
    )

@tool("Zapier Natural Language Actions")
def zapier_nla_tool(action_prompt: str) -> str:
    """
    Args:
        action_prompt: A natural language command (e.g. 'Send a Slack message to #general saying project is live.')
    Returns:
        String confirming the action result or simplified output from Zapier.
    """
    try:
        zapier = ZapierNLAWrapper(zapier_nla_api_key=ZAPIER_NLA_API_KEY)
        return zapier.run(action_prompt)
    except ImportError:
        return "Error: Install dependencies with: pip install langchain-community"
    except Exception as e:
        return f"Error using Zapier NLA Tool: {str(e)}"

def create_zapier_agent(llm):
    return Agent(
        role="Business Automation Specialist",
        goal="Automate business workflows using thousands of apps via Zapier Natural Language Actions",
        backstory=(
            "You are an automation expert who uses natural language to command apps like Gmail, Slack, Trello, and Salesforce via Zapier. "
            "You help users accomplish real work efficiently by connecting and automating their favorite apps."
        ),
        tools=[zapier_nla_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

def create_automation_task(prompt="Send an email to my manager saying the project milestone was completed today."):
    return Task(
        description=(
            f"Use the Zapier Natural Language Actions tool to: '{prompt}'. Confirm once the action is completed."
        ),
        expected_output=(
            f"A summary confirming that the action was completed, with any response or confirmation Zapier provides."
        ),
        agent=None
    )

def check_dependencies():
    missing_deps = []
    try:
        from langchain_community.utilities.zapier import ZapierNLAWrapper
    except ImportError:
        missing_deps.append("langchain-community")
    return missing_deps

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable.")
        return
    if not ZAPIER_NLA_API_KEY:
        print("⚠️  Please set your ZAPIER_NLA_API_KEY environment variable (get from https://nla.zapier.com/docs/authentication/).")
        return
    missing = check_dependencies()
    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Install with: pip install langchain-community")
        return

    print("🚀 Starting Zapier Automation with Gemini 2.5 Flash...")

    # Setup Gemini LLM
    llm = setup_gemini_llm()

    # Create Zapier agent
    agent = create_zapier_agent(llm)

    # Create task
    task = create_automation_task("Send a Slack message to #general saying the deployment succeeded!")
    task.agent = agent

    # Create and run crew
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    print("\n" + "="*50)
    print("ZAPIER NLA ACTION RESULT")
    print("="*50)
    print(result)

def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()
