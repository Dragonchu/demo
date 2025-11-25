from typing import List, Optional
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    name: str = Field(description="Agent 的名称 (例如: 'CodeWriter', 'Reviewer')")
    role: str = Field(description="Agent 的角色描述")
    system_prompt: str = Field(description="定义 Agent 人设和指令的详细系统提示词")

class SystemPlan(BaseModel):
    """多 Agent 系统的整体规划方案"""
    agents: List[AgentConfig] = Field(description="需要创建的子 Agent 列表")
    plan_overview: str = Field(description="这些 Agent 如何协作的简要概述")
