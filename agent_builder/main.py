import sys
import argparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 导入我们的模块
# 使用相对导入进行模块执行
try:
    from .planner import create_agent_plan
    from .builder import build_dynamic_graph
except ImportError:
    # 直接脚本执行的回退
    from planner import create_agent_plan
    from builder import build_dynamic_graph

# 加载环境变量
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="动态多 Agent 系统生成器")
    parser.add_argument("--prompt", type=str, help="你想要的 Agent 系统的自然语言描述。")
    args = parser.parse_args()

    user_request = args.prompt
    if not user_request:
        print("请使用 --prompt 提供提示词")
        # 如果未提供提示词，则使用默认值进行测试
        user_request = "我想要一个能够读取包含代码的文件，分析其安全问题，并将报告写入新文件的系统。"
        print(f"使用默认提示词: {user_request}")

    print(f"\n--- 第一步: 为 '{user_request}' 规划 Agent 系统 ---")
    plan = create_agent_plan(user_request)
    
    print(f"\n已生成计划:")
    print(f"概览: {plan.plan_overview}")
    print(f"Agents: {[a.name for a in plan.agents]}")
    for agent in plan.agents:
        print(f"  - {agent.name}: {agent.role}")

    print(f"\n--- 第二步: 构建动态图 ---")
    graph = build_dynamic_graph(plan)
    print("图构建成功。")

    print(f"\n--- 第三步: 执行系统 ---")
    # 使用用户请求进行初始化
    initial_state = {
        "messages": [HumanMessage(content=user_request)]
    }
    
    # 运行图
    # 我们流式传输输出以查看进度
    try:
        for s in graph.stream(initial_state):
            if "__end__" not in s:
                print(s)
                print("----")
    except Exception as e:
        print(f"执行失败: {e}")
        print("注意: 这需要有效的 OPENAI_API_KEY。")

if __name__ == "__main__":
    main()
