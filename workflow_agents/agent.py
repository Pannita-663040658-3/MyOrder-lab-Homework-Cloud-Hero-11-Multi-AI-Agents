import os
import logging
import google.cloud.logging

from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from google.adk.tools import exit_loop

# Cloud Logging Setup
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

# Model selection from env or default
model_name = os.getenv("MODEL")

# --- Tools ---

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key."""
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}

def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """Saves the final verdict to a .txt file."""
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "success"}

# --- Agents ---

# Step 1: Agent A (The Admirer) - Focused on Positives/Achievements
admirer = Agent(
    name="admirer",
    model=model_name,
    description="Dedicated researcher for positive milestones, breakthroughs, and accolades.",
    instruction="""
    OBJECTIVE: Build a comprehensive portfolio of success for the given TOPIC: { TOPIC? }.
    
    SEARCH STRATEGY:
    - You must modify your Wikipedia search queries to pivot toward excellence. 
    - Always append qualitative descriptors such as "awards", "innovations", "breakthroughs", or "major contributions" to the TOPIC.
    - Example: Instead of just searching for 'SpaceX', search for 'SpaceX achievements and successful launches'.

    DATA HANDLING:
    - Extract specific evidence of impact, legacy, and honors.
    - Use 'append_to_state' to log all verified positive findings into the 'pos_data' field.
    - Conclude with a high-level summary of why this topic is celebrated.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state
    ]
)

# Step 2: Agent B (The Critic) - Focused on Failures/Controversies
critic_investigator = Agent(
    name="critic_investigator",
    model=model_name,
    description="Critical analyst tasked with uncovering disputes, setbacks, and public backlash.",
    instruction="""
    OBJECTIVE: Conduct a rigorous investigation into the risks and failures of the TOPIC: { TOPIC? }.
    
    SEARCH STRATEGY:
    - You are required to hunt for friction. Modify your Wikipedia search queries by adding keywords like "criticism", "legal issues", "failures", "ethical concerns", or "disputes".
    - Example: Instead of searching for 'Artificial Intelligence', search for 'Artificial Intelligence ethical controversies and risks'.

    DATA HANDLING:
    - Document specific instances of failure, negative reception, or institutional mistakes.
    - Use 'append_to_state' to store these findings strictly within the 'neg_data' field.
    - Provide a summary highlighting the primary points of contention or historical errors.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state
    ]
)

# Step 2: Parallel Setup for Investigation
investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer, critic_investigator]
)

# Step 3: Agent C (The Judge) - Balance and Quality Controller
judge = Agent(
    name="judge",
    model=model_name,
    description="Quality assurance officer who ensures objective balance and data depth.",
    instruction="""
    ROLE: You are an impartial Auditor. Your goal is to ensure the dossier is balanced and intellectually robust.

    DATA EVALUATION:
    - Positive Evidence (POS_DATA): { pos_data? }
    - Negative Evidence (NEG_DATA): { neg_data? }

    AUDIT PROTOCOL:
    1. **Symmetry Check**: Compare the volume and depth of POS_DATA vs NEG_DATA. If one side is significantly thinner or contains only surface-level info, it is 'Incomplete'.
    2. **Granularity Check**: Does the data contain specific facts/dates? If it is too vague, it needs a deeper search.

    DECISION LOGIC:
    - IF UNSATISFACTORY: Identify exactly which side is lacking. Issue a 'RE-RESEARCH' command. Provide a specific directive for the Researcher, such as: "Search for [Topic] + specific controversies regarding [X]" or "Find more technical milestones for [Topic]". DO NOT call exit_loop.
    - IF SATISFACTORY: If both sides are equally dense and provide a 360-degree view, call the 'exit_loop' tool to conclude the session.
    """,
    tools=[exit_loop]
)

# Step 3: Loop Setup for Iterative Review
trial_review_loop = LoopAgent(
    name="trial_review_loop",
    sub_agents=[investigation_team, judge],
    max_iterations=3
)

# Step 4: The Verdict Agent - Final Report Generation
verdict_agent = Agent(
    name="verdict_agent",
    model=model_name,
    description="Strategic analyst that synthesizes multi-perspective data into a formal executive report.",
    instruction="""
    OBJECTIVE: Consolidate the dual-perspective research into a definitive, balanced, and high-quality analysis.

    DATA SOURCES:
    - Positive Evidence Vault: { pos_data? }
    - Critical Analysis Vault: { neg_data? }

    WRITING GUIDELINES:
    - Tone: Objective, analytical, and strictly neutral (avoid bias).
    - Synthesis: Do not just list points; contrast the achievements against the challenges to provide a 360-degree view.

    TASK:
    1. Generate a structured report using the following format:
       ## EXECUTIVE DOSSIER: [TOPIC NAME]
       ### I. CONTEXTUAL OVERVIEW: Brief background of the subject.
       ### II. MILESTONES & ACHIEVEMENTS: In-depth analysis of positive contributions.
       ### III. DISPUTES & LIMITATIONS: Examination of controversies and critical setbacks.
       ### IV. COMPREHENSIVE SYNTHESIS: A balanced final statement weighing both sides.

    2. FILE EXPORT:
       - Tool: Use 'write_file' to commit this report to disk.
       - Path: Save within the 'court_reports' directory.
       - Filename: Cleaned TOPIC name (e.g., 'topic_name.txt').
    """,
    tools=[write_file]
)

# Sequential Workflow (Step 2 to 4)
historical_court_system = SequentialAgent(
    name="historical_court_system",
    description="Main workflow for the Historical Court analysis.",
    sub_agents=[
        trial_review_loop,
        verdict_agent
    ]
)

# Step 1: The Inquiry - Entry Point
root_agent = Agent(
    name="inquiry_agent",
    model=model_name,
    description="Welcomes the user and receives the historical topic.",
    instruction="""
    INSTRUCTIONS:
    - Welcome the user to 'The Historical Court' MAS.
    - Ask the user for the name of a historical figure or event they wish to analyze.
    - When the user provides a topic, use 'append_to_state' to store it in the 'TOPIC' key.
    - Transfer control to the 'historical_court_system'.
    """,
    tools=[append_to_state],
    sub_agents=[historical_court_system]
)