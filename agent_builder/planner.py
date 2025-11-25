import os
from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from .models import SystemPlan, AgentConfig

class MockChatModel(BaseChatModel):
    """一个用于测试的模拟聊天模型，返回预定义的响应。"""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return getattr(self, "response", AIMessage(content="模拟响应"))

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"
    
    def bind_functions(self, *args, **kwargs):
        return self
        
    def bind_tools(self, *args, **kwargs):
        return self

def get_llm():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("警告: 未找到 OPENAI_API_KEY。正在使用模拟 LLM。")
        return MockChatModel()
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

def create_agent_plan(user_request: str) -> SystemPlan:
    llm = get_llm()
    
    # 如果我们使用模拟模型，直接返回硬编码的计划
    if isinstance(llm, MockChatModel):
        return SystemPlan(
            plan_overview="模拟计划: 安全分析系统",
            agents=[
                AgentConfig(name="FileReader", role="读取文件", system_prompt="你使用 read_file 工具读取文件。"),
                AgentConfig(name="SecurityAnalyst", role="分析代码", system_prompt="你使用 code_analysis 工具分析代码。"),
                AgentConfig(name="ReportWriter", role="编写报告", system_prompt="你使用 write_file 工具编写报告。")
            ]
        )

    system_prompt = """你是一位专家级的系统架构师。
    你的目标是分析用户对多 Agent 系统的需求，并将其拆解为具体的子 Agent。
    
    系统架构采用 Supervisor/Controller 模式，由中心控制器管理这些子 Agent。
    
    对于每个需要的子 Agent，请确定：
    1. 唯一的名称 (name)。
    2. 具体的角色 (role)。
    3. 详细的系统提示词 (system_prompt)，指导 Agent 如何执行任务以及何时使用工具。
    
    所有 Agent 都将获得统一的工具集（文件读写、搜索、代码分析等），
    你不需要分配工具，但应在 system_prompt 中指导它们在适当时机使用工具。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "用户需求: {request}")
    ])
    
    chain = prompt | llm.with_structured_output(SystemPlan)
    
    return chain.invoke({"request": user_request})
