from typing import List, Optional
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    name: str = Field(description="The name of the agent (e.g., 'CodeWriter', 'Reviewer').")
    role: str = Field(description="A description of the agent's role.")
    system_prompt: str = Field(description="The system prompt that defines the agent's persona and instructions.")

class SystemPlan(BaseModel):
    """The plan for the multi-agent system."""
    agents: List[AgentConfig] = Field(description="The list of sub-agents to create.")
    plan_overview: str = Field(description="A brief overview of how these agents interact.")
