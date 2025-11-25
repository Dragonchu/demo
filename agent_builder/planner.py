import os
from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from .models import SystemPlan, AgentConfig

class MockChatModel(BaseChatModel):
    """A mock chat model that returns predefined responses for testing."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return getattr(self, "response", AIMessage(content="Mock response"))

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
        print("Warning: No OPENAI_API_KEY found. Using Mock LLM.")
        return MockChatModel()
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

def create_agent_plan(user_request: str) -> SystemPlan:
    llm = get_llm()
    
    # If we are using the mock, return a hardcoded plan directly
    if isinstance(llm, MockChatModel):
        return SystemPlan(
            plan_overview="Mock Plan: Security Analysis System",
            agents=[
                AgentConfig(name="FileReader", role="Reads files", system_prompt="You read files using the read_file tool."),
                AgentConfig(name="SecurityAnalyst", role="Analyzes code", system_prompt="You analyze code using code_analysis tool."),
                AgentConfig(name="ReportWriter", role="Writes reports", system_prompt="You write reports using write_file tool.")
            ]
        )

    system_prompt = """You are an expert system architect. 
    Your goal is to analyze the user's request for a multi-agent system and break it down into specific sub-agents.
    
    The system architecture will be a Supervisor/Controller model where a central controller manages these sub-agents.
    
    For each required sub-agent, determine:
    1. A unique name.
    2. A specific role.
    3. A detailed system prompt that instructs the agent on how to perform its specific task and how to use tools if necessary.
    
    All agents will have access to a shared set of tools (File I/O, Search, Code Analysis), so you don't need to assign tools, 
    but you should instruct them in the system prompt to use them when appropriate.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Request: {request}")
    ])
    
    chain = prompt | llm.with_structured_output(SystemPlan)
    
    return chain.invoke({"request": user_request})
