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

# 1. Define the shared state
class AgentState(TypedDict):
    # The annotation tells the graph that new messages are appended to the existing list
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # We could add 'next' here to track who goes next
    next: str

# 2. Helper to create a sub-agent node
def create_agent_node(agent_config, llm, tools):
    """
    Creates a node for the graph that runs a specific agent.
    We use create_react_agent for the standard tool-using loop within the node.
    """
    
    # Check for Mock LLM
    if hasattr(llm, "_llm_type") and llm._llm_type == "mock-chat-model":
        def mock_agent_node(state):
            # Return a mock response
            return {"messages": [AIMessage(content=f"[{agent_config.name}] Mock work done.")]}
        return mock_agent_node

    # The agent needs a system message. 
    # create_react_agent takes 'state_modifier' or 'messages_modifier' to inject system prompt.
    
    # We create the compiled graph for the agent
    agent_runnable = create_react_agent(llm, tools, state_modifier=agent_config.system_prompt)
    
    def agent_node(state):
        # The agent runnable takes the state and returns a dictionary with updates
        # We invoke the agent with the current state
        result = agent_runnable.invoke(state)
        
        # Calculate new messages to avoid duplication in the parent graph
        # because parent graph uses operator.add for messages.
        original_count = len(state["messages"])
        all_messages = result["messages"]
        new_messages = all_messages[original_count:]
        
        return {"messages": new_messages}

    return agent_node

# 3. The Supervisor Node
def create_supervisor_node(llm, members: List[str]):
    """
    Creates the supervisor node that decides which agent acts next.
    """
    
    # Mock Supervisor Logic
    if hasattr(llm, "_llm_type") and llm._llm_type == "mock-chat-model":
        def mock_supervisor(state):
            # Simple round-robin or random logic for demo
            # Or just call each once then finish.
            messages = state.get("messages", [])
            last_sender = "user"
            if messages and isinstance(messages[-1], AIMessage):
                content = messages[-1].content
                # Extract sender from content if we formatted it that way, 
                # but usually we check the name field if set, or just infer.
                # For this mock, let's just iterate through members.
                
            # Find who hasn't run?
            # Let's just return the first member, then the second...
            # This is hard to do stateless. 
            # We'll just randomize or pick the first one that hasn't "spoken" in the last N messages?
            
            # Simplification: If < 5 messages, pick a random member, else FINISH
            import random
            if len(messages) < 5:
                return {"next": random.choice(members)}
            return {"next": "FINISH"}
        return mock_supervisor

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    
    options = ["FINISH"] + members
    
    # Using OpenAI function calling for structured routing
    function_def = {
        "name": "route",
        "description": "Select the next role.",
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
                "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    
    def supervisor_node(state):
        # Invokes the router logic
        result = supervisor_chain.invoke(state)
        return result # returns {"next": "AgentName"} or {"next": "FINISH"}

    return supervisor_node

# 4. Main Build Function
def build_dynamic_graph(plan: SystemPlan):
    """
    Constructs the LangGraph based on the plan.
    """
    llm = get_llm()
    tools = ALL_TOOLS
    
    members = [agent.name for agent in plan.agents]
    workflow = StateGraph(AgentState)
    
    # Add Supervisor
    supervisor_node = create_supervisor_node(llm, members)
    workflow.add_node("supervisor", supervisor_node)
    
    # Add Worker Agents
    for agent_cfg in plan.agents:
        node_func = create_agent_node(agent_cfg, llm, tools)
        workflow.add_node(agent_cfg.name, node_func)
        
        # Edge: Worker -> Supervisor
        # After a worker is done, it goes back to supervisor
        workflow.add_edge(agent_cfg.name, "supervisor")
        
    # Edges: Supervisor -> Workers (Conditional)
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"], 
        conditional_map
    )
    
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()
