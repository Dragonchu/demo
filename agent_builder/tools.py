from langchain_core.tools import tool

@tool
def read_file(file_path: str) -> str:
    """读取文件内容。"""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"读取文件错误: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """将内容写入文件。"""
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return f"成功写入到 {file_path}"
    except Exception as e:
        return f"写入文件错误: {e}"

@tool
def code_analysis(code_snippet: str) -> str:
    """分析给定的代码片段是否存在潜在问题。"""
    return f"分析完成: 代码结构看似合理但缺乏注释。(模拟分析: {code_snippet[:20]}...)"

@tool
def web_search(query: str) -> str:
    """执行网络搜索。"""
    return f"'{query}' 的搜索结果: [模拟结果 1], [模拟结果 2]"

# 所有可用工具列表
ALL_TOOLS = [read_file, write_file, code_analysis, web_search]
