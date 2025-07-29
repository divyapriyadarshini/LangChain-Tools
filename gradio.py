# runtime error from Hugging Face Spaces, issue within the tool
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from gradio_tools.tools import (
    StableDiffusionTool,
    ImageCaptioningTool,
    StableDiffusionPromptGeneratorTool,
    TextToVideoTool
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def setup_gemini_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )

@tool("Stable Diffusion Image Generator")
def generate_image(prompt: str) -> str:
    """
    Generate an image using Stable Diffusion based on text prompt.
    Args:
        prompt: Text description of the image to generate
    Returns:
        Local file path of the generated image
    """
    try:
        stable_diffusion = StableDiffusionTool()
        return stable_diffusion.langchain.run(prompt)
    except Exception as e:
        return f"Error generating image: {str(e)}"

@tool("Image Caption Generator")
def caption_image(image_path: str) -> str:
    """
    Generate a caption for an image.
    Args:
        image_path: Path to the image file
    Returns:
        Text caption describing the image
    """
    try:
        captioning_tool = ImageCaptioningTool()
        return captioning_tool.langchain.run(image_path)
    except Exception as e:
        return f"Error captioning image: {str(e)}"

@tool("Stable Diffusion Prompt Improver")
def improve_prompt(prompt: str) -> str:
    """
    Improve a text prompt for better Stable Diffusion results.
    Args:
        prompt: Basic text prompt to improve
    Returns:
        Enhanced prompt with artistic style descriptors
    """
    try:
        prompt_generator = StableDiffusionPromptGeneratorTool()
        return prompt_generator.langchain.run(prompt)
    except Exception as e:
        return f"Error improving prompt: {str(e)}"

@tool("Text to Video Generator")
def generate_video(description: str) -> str:
    """
    Generate a video from text description.
    Args:
        description: Text description of the video content
    Returns:
        Local file path of the generated video
    """
    try:
        video_tool = TextToVideoTool()
        return video_tool.langchain.run(description)
    except Exception as e:
        return f"Error generating video: {str(e)}"

def create_gradio_agent(llm, tools):
    return Agent(
        role="AI Creative Assistant",
        goal="Use Gradio tools to create, enhance, and process multimedia content based on user requests",
        backstory="You are an AI assistant specialized in using various Gradio tools for image generation, captioning, prompt enhancement, and video creation.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_gradio_task(user_request, tool_args_dict):
    tool_args_str = "\n".join([f"- {k}: {v}" for k, v in tool_args_dict.items()]) if tool_args_dict else "None"
    
    task_description = f"""
    You have access to multiple Gradio tools for multimedia content creation and processing.

    ### User Request:
    {user_request}

    ### Tool Parameters:
    {tool_args_str}

    Please:
    1. Analyze the user request to determine which tool(s) to use
    2. Use the provided parameters or extract them from the user request
    3. Execute the appropriate tool(s) in the correct sequence
    4. Return the final output with file paths or descriptions as appropriate

    Available tools:
    - Stable Diffusion Image Generator: Creates images from text prompts
    - Image Caption Generator: Generates captions for images
    - Stable Diffusion Prompt Improver: Enhances prompts for better image generation
    - Text to Video Generator: Creates videos from text descriptions
    """
        
    return Task(
        description=task_description,
        expected_output="Complete multimedia processing result with file paths and descriptions",
        agent=None  # to be assigned later
    )


def get_user_input():
    print("Gradio AI Creative Tools")
    print("=" * 30)
    print("Available operations:")
    print("1. Generate image from text")
    print("2. Caption existing image")
    print("3. Improve image prompt")
    print("4. Generate video from text")
    print("5. Full workflow (improve prompt → generate image → caption → create video)")
    
    choice = input("\nSelect operation (1-5): ").strip()
    user_request = ""
    tool_args = {}
    
    if choice == "1":
        prompt = input("Enter image description: ").strip()
        user_request = f"Generate an image based on: {prompt}"
        tool_args = {"prompt": prompt}
    elif choice == "2":
        image_path = input("Enter image file path: ").strip()
        user_request = f"Generate a caption for the image at: {image_path}"
        tool_args = {"image_path": image_path}
    elif choice == "3":
        prompt = input("Enter basic prompt to improve: ").strip()
        user_request = f"Improve this prompt for better image generation: {prompt}"
        tool_args = {"prompt": prompt}
    elif choice == "4":
        description = input("Enter video description: ").strip()
        user_request = f"Generate a video based on: {description}"
        tool_args = {"description": description}
    elif choice == "5":
        prompt = input("Enter basic image idea: ").strip()
        user_request = f"Create a complete workflow: improve the prompt '{prompt}', generate an image, caption it, and create a video"
        tool_args = {"initial_prompt": prompt}
    else:
        prompt = input("Enter your creative request: ").strip() or "a dog riding a skateboard"
        user_request = prompt
        tool_args = {"prompt": prompt}
    
    return user_request, tool_args

def main():
    if not GEMINI_API_KEY:
        print("⚠️ Please set your GEMINI_API_KEY environment variable")
        return
    
    user_request, tool_args = get_user_input()
    
    print(f"\n🚀 Starting CrewAI Gradio Tools Processing")
    print(f"Request: {user_request}")
    
    tools = [
        generate_image,
        caption_image,
        improve_prompt,
        generate_video
    ]
    
    llm = setup_gemini_llm()
    agent = create_gradio_agent(llm, tools)
    task = create_gradio_task(user_request, tool_args)
    task.agent = agent
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("\n🎨 Processing with Gradio tools...")
    print("=" * 50)
    result = crew.kickoff()
    print("\n📁 Final Result:")
    print(result)

if __name__ == "__main__":
    main()
