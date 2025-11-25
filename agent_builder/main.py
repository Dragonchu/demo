import sys
import argparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import our modules
# Use relative imports for module execution
try:
    from .planner import create_agent_plan
    from .builder import build_dynamic_graph
except ImportError:
    # Fallback for direct script execution
    from planner import create_agent_plan
    from builder import build_dynamic_graph

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Dynamic Multi-Agent System Generator")
    parser.add_argument("--prompt", type=str, help="The natural language description of the agent system you want.")
    args = parser.parse_args()

    user_request = args.prompt
    if not user_request:
        print("Please provide a prompt using --prompt")
        # Default for testing if no prompt provided
        user_request = "I want a system that can read a file containing code, analyze it for security issues, and write a report to a new file."
        print(f"Using default prompt: {user_request}")

    print(f"\n--- Step 1: Planning Agent System for: '{user_request}' ---")
    plan = create_agent_plan(user_request)
    
    print(f"\nPlan Generated:")
    print(f"Overview: {plan.plan_overview}")
    print(f"Agents: {[a.name for a in plan.agents]}")
    for agent in plan.agents:
        print(f"  - {agent.name}: {agent.role}")

    print(f"\n--- Step 2: Building Dynamic Graph ---")
    graph = build_dynamic_graph(plan)
    print("Graph built successfully.")

    print(f"\n--- Step 3: Executing System ---")
    # Initialize with the user request
    initial_state = {
        "messages": [HumanMessage(content=user_request)]
    }
    
    # Run the graph
    # We stream the output to see progress
    try:
        for s in graph.stream(initial_state):
            if "__end__" not in s:
                print(s)
                print("----")
    except Exception as e:
        print(f"Execution failed: {e}")
        print("Note: This requires a valid OPENAI_API_KEY.")

if __name__ == "__main__":
    main()
