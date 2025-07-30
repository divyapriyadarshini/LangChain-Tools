# requires WAITLIST for API key
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from ads4gpts_langchain import Ads4gptsInlineSponsoredResponseTool, Ads4gptsToolkit

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADS4GPTS_API_KEY = os.getenv("ADS4GPTS_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("ADS4GPTs Inline Sponsored Response")
def ads_inline_sponsored_response(
    id: str,
    user_gender: str,
    user_age: str,
    user_persona: str,
    ad_recommendation: str,
    undesired_ads: str,
    context: str,
    num_ads: int = 1,
    style: str = "neutral",
    ad_format: str = "INLINE_SPONSORED_RESPONSE"
) -> str:
    """
    Fetches native, sponsored responses for ad placement.
    """
    try:
        tool = Ads4gptsInlineSponsoredResponseTool(ads4gpts_api_key=ADS4GPTS_API_KEY)
        result = tool._run(
            id=id,
            user_gender=user_gender,
            user_age=user_age,
            user_persona=user_persona,
            ad_recommendation=ad_recommendation,
            undesired_ads=undesired_ads,
            context=context,
            num_ads=num_ads,
            style=style,
            ad_format=ad_format
        )
        return f"Inline Sponsored Response Result:\n{result['ad_text']}"
    except Exception as e:
        return f"Error getting sponsored response: {str(e)}"

def create_ads_agent(llm, tools):
    return Agent(
        role="AI Native Advertising Specialist",
        goal="Monetize AI applications by fetching and integrating contextually-relevant ads using ADS4GPTs tools",
        backstory="You are fluent in native advertising for AI interfaces, skilled at selecting and integrating ads with privacy and user experience in mind.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_ads_task(ad_request, ad_params):
    params_str = "\n".join([f"- {k}: {v}" for k, v in ad_params.items()]) if ad_params else "None"
    task_description = f"""
You have access to ADS4GPTs native ad tools for AI monetization.

### Ad Request:
{ad_request}

### Ad Parameters:
{params_str}

Please use "ADS4GPTs Inline Sponsored Response" for fetching and integrating a native ad tailored to the provided context and user attributes.
"""
    return Task(
        description=task_description,
        expected_output="Inline sponsored advertising message or appropriate error message.",
        agent=None  
    )

def get_user_input():
    print("ADS4GPTs Native Ad Tool")
    print("=" * 30)
    user_gender = input("User gender: ").strip() or "female"
    user_age = input("User age (e.g. 25-34): ").strip() or "25-34"
    user_persona = input("User persona: ").strip() or "test_persona"
    ad_recommendation = input("Desired ad topic: ").strip() or "latest fashion sale"
    undesired_ads = input("Blacklisted ad types: ").strip() or "gambling"
    context = input("Context (user scenario): ").strip() or "user browsing for summer wear"
    num_ads = int(input("Number of ads (default 1): ").strip() or "1")
    style = input("Ad style (neutral, youthful, etc): ").strip() or "neutral"
    ad_format = "INLINE_SPONSORED_RESPONSE"
    ad_request = f"Fetch an inline sponsored ad for a '{user_gender}', age '{user_age}', interested in '{ad_recommendation}'"
    ad_params = {
        "id": "unique_user_id_001", "user_gender": user_gender, "user_age": user_age,
        "user_persona": user_persona, "ad_recommendation": ad_recommendation,
        "undesired_ads": undesired_ads, "context": context,
        "num_ads": num_ads, "style": style, "ad_format": ad_format
    }
    return ad_request, ad_params

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    if not ADS4GPTS_API_KEY:
        print("⚠️ Please set your ADS4GPTS_API_KEY environment variable")
        print("   1. Get your API key from https://ads4gpts.com/dashboard")
        return
    ad_request, ad_params = get_user_input()
    print(f"\n🚀 Starting ADS4GPTs Ad Placement: {ad_request}")
    tools = [ads_inline_sponsored_response]
    llm = setup_gemini_llm()
    agent = create_ads_agent(llm, tools)
    task = create_ads_task(ad_request, ad_params)
    task.agent = agent

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    print("\n💰 Running ADS4GPTs ad tool...\n" + "="*50)
    result = crew.kickoff()
    print("\n📢 Ad Result:\n" + "="*50)
    print(result)

if __name__ == "__main__":
    main()
