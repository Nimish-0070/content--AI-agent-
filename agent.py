# ============================================================
# agent.py — Stable CrewAI Multi-Agent + Fallback Pipeline
# ============================================================

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dot_env = load_dotenv()

# Load API keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchTool

# Gemini API
from google import genai
genai_client = genai.Client(api_key=GEMINI_KEY)


# ============================================================
# Gemini Wrapper (Used by all Agents + Fallback)
# ============================================================
class GeminiLLM:
    def __init__(self, model="gemini-2.5-flash"):
        self.model = model

    def run(self, prompt: str) -> str:
        """
        Gemini Smart Retry + Model Fallback
        Only uses VALID models for google-genai SDK.
        """

        valid_models = [
            "gemini-2.5-flash",   # main
            "gemini-2.0-pro",     # fallback 1
            "gemini-2.0-flash",   # fallback 2
        ]

        for model_name in valid_models:
            try:
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if hasattr(response, "text") and response.text:
                    return response.text

                try:
                    return response.candidates[0].content.parts[0].text
                except:
                    return str(response)

            except Exception as e:
                # If model unavailable or overloaded → try next
                if "503" in str(e) or "UNAVAILABLE" in str(e) or "404" in str(e):
                    continue
                return f"[Gemini Error] {e}"

        # FINAL BACKUP
        return (
            "⚠ **Gemini servers are busy. Fallback activated.**\n\n"
            f"Prompt preview: {prompt[:120]}...\n\n"
            "- Please try again after a few seconds.\n"
            "- This is offline fallback content."
        )




llm = GeminiLLM()


# ============================================================
# Tavily Search Tool
# ============================================================
tavily_tool = TavilySearchTool(api_key=TAVILY_KEY)


# ============================================================
# CrewAI Agents
# ============================================================
research_agent = Agent(
    name="Research Agent",
    role="Research Specialist",
    goal="Collect verified information using Tavily search.",
    backstory="A professional researcher who extracts the most relevant details.",
    tools=[tavily_tool],
    llm=llm,
)

writer_agent = Agent(
    name="Writer Agent",
    role="Content Writer",
    goal="Write high-quality structured content.",
    backstory="A professional writer with strong SEO knowledge.",
    llm=llm,
)

editor_agent = Agent(
    name="Editor Agent",
    role="Editor",
    goal="Refine, improve clarity, and polish writing.",
    backstory="A grammar-driven expert editor.",
    llm=llm,
)

seo_agent = Agent(
    name="SEO Agent",
    role="SEO Specialist",
    goal="Generate SEO-friendly keywords, tags, and metadata.",
    backstory="Expert in making content rank on search engines.",
    llm=llm,
)


# ============================================================
# Safe Crew Runner — handles API version differences
# ============================================================
def run_crew_safely(crew_obj):
    """Attempts all known CrewAI run methods."""
    method_names = ["run", "execute", "start", "launch", "run_all", "run_pipeline"]

    for method in method_names:
        fn = getattr(crew_obj, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass

    raise RuntimeError("No compatible CrewAI run method found.")


# ============================================================
# Fallback local pipeline (Gemini + Tavily)
# ============================================================
def fallback_pipeline(topic, content_type, length, tone):
    """Used when CrewAI fails."""

    # Try research
    research_results = []
    try:
        response = tavily_tool.search(topic)
        research_results = response.get("results", [])
    except:
        research_results = []

    research_text = "\n".join(str(r) for r in research_results)

    # Compose main content prompt
    prompt = f"""
Write a {content_type} on the topic: {topic}
Tone: {tone}
Length: {length}

Use this research if available:

{research_text}

Structure it with:
- Title
- Meta description
- Introduction
- 4–6 subheadings
- Conclusion
- SEO keywords
"""

    generated = llm.run(prompt)

    # Polishing content
    polished = llm.run(f"Polish and refine this content:\n\n{generated}")

    # Basic SEO keyword extraction
    try:
        words = [
            w.strip(".,!?()").lower()
            for w in polished.split()
            if len(w) > 3
        ]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        keywords = ", ".join(list(freq.keys())[:10])
    except:
        keywords = "content, ai, automation"

    pipeline = {
        "ResearchAgent": research_text if research_text else "No research available",
        "WriterAgent": generated,
        "EditorAgent": polished,
        "SEOAgent": keywords
    }

    return polished, pipeline


# ============================================================
# MAIN FUNCTION USED BY STREAMLIT
# ============================================================
def create_content(topic: str, content_type: str, length: str, tone: str, use_research: bool) -> Dict[str, Any]:

    # Crew Tasks
    research_task = Task(
        description=f"Research the topic '{topic}'.",
        agent=research_agent,
        expected_output="Detailed research summary."
    )

    writing_task = Task(
        description=f"Write a {content_type} about '{topic}', length {length}, tone {tone}.",
        agent=writer_agent,
        expected_output="Structured content draft."
    )

    editing_task = Task(
        description="Polish and edit the draft.",
        agent=editor_agent,
        expected_output="Refined content."
    )

    seo_task = Task(
        description="Generate SEO keywords and metadata.",
        agent=seo_agent,
        expected_output="SEO suggestions."
    )

    crew = Crew(
        agents=[research_agent, writer_agent, editor_agent, seo_agent],
        tasks=[research_task, writing_task, editing_task, seo_task],
        process=Process.sequential,
        verbose=True
    )

    # TRY CREWAI
    try:
        result = run_crew_safely(crew)

        return {
            "success": True,
            "final_output": result,
            "pipeline": crew.tasks_output
        }

    # ON FAILURE → FALLBACK
    except Exception as e:
        print("[CrewAI ERROR] Falling back. Error:", e)

        fallback_output, fallback_details = fallback_pipeline(topic, content_type, length, tone)

        return {
            "success": False,
            "error": str(e),
            "final_output": fallback_output,
            "pipeline": fallback_details
        }
