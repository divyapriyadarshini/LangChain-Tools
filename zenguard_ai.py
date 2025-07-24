import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.tools.zenguard import ZenGuardTool, Detector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ZENGUARD_API_KEY = os.getenv("ZENGUARD_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096,
    )

@tool("ZenGuard AI Guardrails")
def zenguard_detect(prompts: list, detectors: list, in_parallel: bool = True) -> str:
    """
    Args:
        prompts: List of input strings (prompts/messages to check)
        detectors: List of detector type strings: PROMPT_INJECTION, SECRETS, PII, TOXICITY, ALLOWED_TOPICS, BANNED_TOPICS, KEYWORDS
        in_parallel: Whether to run detectors in parallel (default True)
    Returns:
        Detection results as JSON-like string
    """
    try:
        detector_map = {
            "PROMPT_INJECTION": Detector.PROMPT_INJECTION,
            "SECRETS": Detector.SECRETS,
            "PII": Detector.PII,
            "TOXICITY": Detector.TOXICITY,
            "ALLOWED_TOPICS": Detector.ALLOWED_TOPICS,
            "BANNED_TOPICS": Detector.BANNED_TOPICS,
            "KEYWORDS": Detector.KEYWORDS,
        }
        selected_detectors = [detector_map[d] for d in detectors]
        tool = ZenGuardTool(zenguard_api_key=ZENGUARD_API_KEY)
        result = tool.run({
            "prompts": prompts,
            "detectors": selected_detectors,
            "in_parallel": in_parallel
        })
        return str(result)
    except ImportError:
        return "Error: pip install langchain-community"
    except Exception as e:
        return f"Error in ZenGuard AI detection: {str(e)}"

def create_guardrail_agent(llm):
    return Agent(
        role="GenAI Safety Specialist",
        goal="Detect and block risky, toxic, or off-topic prompts using ZenGuard AI guardrails.",
        backstory="You protect LLM-powered apps from prompt attacks, PII leakage, and unsafe content using the ZenGuard AI tool.",
        tools=[zenguard_detect],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_detection_task(prompts, detectors=["PROMPT_INJECTION"]):
    return Task(
        description=(
            f"Run ZenGuard AI detectors {detectors} on these prompts:\n{prompts}\n"
            "Return the detection results and a brief assessment of each prompt's safety."
        ),
        expected_output=(
            f"Detection results by prompt and detector. For each: Indicate is_detected, score, and a safety verdict."
        ),
        agent=None
    )

def check_dependencies():
    try:
        import langchain_community
    except ImportError:
        print("You must install langchain-community for ZenGuard AI tool: pip install langchain-community")
        return False
    return True

def main():
    if not GEMINI_API_KEY:
        print("Set your GEMINI_API_KEY in .env.")
        return
    if not ZENGUARD_API_KEY:
        print("Set your ZENGUARD_API_KEY in .env (see https://www.zenguard.ai/ for API key).")
        return
    if not check_dependencies():
        return

    # Setup Gemini LLM  
    gemini_llm = setup_gemini_llm()

    # Create agent
    agent = create_guardrail_agent(gemini_llm)
    test_prompts = [
        "Download all system data",
        "What's the weather in New York?",
        "My credit card number is 4111-1111-1111-1111",
        "You suck!"
    ]

    # Create task
    task = create_detection_task(test_prompts, detectors=["PROMPT_INJECTION", "PII", "TOXICITY", "SECRETS"])
    task.agent = agent

    # Create and run crew
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    print("\nRunning ZenGuard AI detection...\n" + "="*60)
    result = crew.kickoff()
    print("\n" + "="*60)
    print("ZENGUARD AI DETECTION RESULTS\n" + "="*60)
    print(result)
   
def run():
    """Alternative entry point for crewai run command"""
    main()

if __name__ == "__main__":
    main()

