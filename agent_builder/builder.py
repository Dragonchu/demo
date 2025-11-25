import functools
import operator
from typing import Sequence, TypedDict, Annotated, List, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent

from .models import SystemPlan
from .tools import ALL_TOOLS
from .planner import get_llm

# 1. 定义共享状态
class AgentState(TypedDict):
    # 注解告诉图，新消息会被追加到现有列表中
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 我们可以在这里添加 'next' 来跟踪下一个是谁
    next: str

# 2. 创建子 Agent 节点的辅助函数
def create_agent_node(agent_config, llm, tools):
    """
    为图创建一个运行特定 Agent 的节点。
    我们在节点内使用 create_react_agent 进行标准的工具使用循环。
    """
    
    # 检查是否为模拟 LLM
    if hasattr(llm, "_llm_type") and llm._llm_type == "mock-chat-model":
        def mock_agent_node(state):
            # 返回模拟响应
            return {"messages": [AIMessage(content=f"[{agent_config.name}] 模拟工作完成。")]}
        return mock_agent_node

    # Agent 需要一个系统消息。
    # create_react_agent 接受 'state_modifier' 或 'messages_modifier' 来注入系统提示词。
    
    # 我们为 Agent 创建编译后的图
    agent_runnable = create_react_agent(llm, tools, state_modifier=agent_config.system_prompt)
    
    def agent_node(state):
        # Agent runnable 接收状态并返回包含更新的字典
        # 我们使用当前状态调用 Agent
        result = agent_runnable.invoke(state)
        
        # 计算新消息以避免在父图中重复
        # 因为父图使用 operator.add 来处理消息。
        original_count = len(state["messages"])
        all_messages = result["messages"]
        new_messages = all_messages[original_count:]
        
        return {"messages": new_messages}

    return agent_node

# 3. Supervisor (控制器) 节点
def create_supervisor_node(llm, members: List[str]):
    """
    创建决定下一个行动 Agent 的 Supervisor 节点。
    """
    
    # 模拟 Supervisor 逻辑
    if hasattr(llm, "_llm_type") and llm._llm_type == "mock-chat-model":
        def mock_supervisor(state):
            # 演示用的简单轮询或随机逻辑
            # 或者每个调用一次然后结束。
            messages = state.get("messages", [])
            
            # 简化：如果消息少于 5 条，随机选择一个成员，否则结束 (FINISH)
            import random
            if len(messages) < 5:
                return {"next": random.choice(members)}
            return {"next": "FINISH"}
        return mock_supervisor

    system_prompt = (
        "你是一个主管，负责管理以下工作人员之间的对话：{members}。"
        "根据用户的请求和当前的对话历史，决定下一个由谁来行动。"
        "每个工作人员会执行任务并返回结果。"
        "当任务全部完成时，回复 FINISH。"
    )
    
    options = ["FINISH"] + members
    
    # 使用 OpenAI 函数调用进行结构化路由
    function_def = {
        "name": "route",
        "description": "选择下一个角色。",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next Role",
                    "anyOf": [
                        {"enum": options}
                    ],
                }
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "基于以上对话，谁应该下一个行动？或者应该结束(FINISH)？选择: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    
    def supervisor_node(state):
        # 调用路由逻辑
        result = supervisor_chain.invoke(state)
        return result # 返回 {"next": "AgentName"} 或 {"next": "FINISH"}

    return supervisor_node

# 4. 主构建函数
def build_dynamic_graph(plan: SystemPlan):
    """
    根据计划构建 LangGraph。
    """
    llm = get_llm()
    tools = ALL_TOOLS
    
    members = [agent.name for agent in plan.agents]
    workflow = StateGraph(AgentState)
    
    # 添加 Supervisor
    supervisor_node = create_supervisor_node(llm, members)
    workflow.add_node("supervisor", supervisor_node)
    
    # 添加 Worker Agents
    for agent_cfg in plan.agents:
        node_func = create_agent_node(agent_cfg, llm, tools)
        workflow.add_node(agent_cfg.name, node_func)
        
        # 边: Worker -> Supervisor
        # Worker 完成后，回到 Supervisor
        workflow.add_edge(agent_cfg.name, "supervisor")
        
    # 边: Supervisor -> Workers (条件边)
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"], 
        conditional_map
    )
    
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()
