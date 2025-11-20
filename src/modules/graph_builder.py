import os
from typing import TypedDict, Annotated, List, Union, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# 导入模块一和模块二
from src.modules.rag_engine import GoldKnowledgeBase
from src.modules.customer_brain import generate_customer_response, CustomerResponse

# ==========================================
# 1. 全局资源初始化
# ==========================================
# 为了避免每次请求都重新加载 RAG，我们在模块加载时初始化单例
# 注意：在实际生产中，这通常放在 Application Context 中管理
rag = GoldKnowledgeBase()
try:
    rag.initialize_knowledge_base()
except Exception as e:
    print(f"Warning: RAG initialization failed: {e}")

# ==========================================
# 2. 定义状态 (State Definition)
# ==========================================
class SalesState(TypedDict):
    """
    定义销售过程中的状态数据结构。
    LangGraph 会自动在节点间传递这个字典。
    """
    # messages: 存储对话历史。add_messages reduce 函数不仅添加新消息，还处理 ID 匹配
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 上下文信息
    customer_persona: str  # 客户性格
    sales_stage: str       # 当前销售阶段
    
    # UI 展示与控制字段
    latest_thought: str    # AI 的思维链 (CoT)，用于前端展示
    status: str            # 对话状态: CONTINUE, DEAL, LEAVE

# ==========================================
# 3. 定义节点逻辑 (Nodes)
# ==========================================
def customer_node(state: SalesState) -> Dict[str, Any]:
    """
    客户节点：接收销售的话，进行思考，生成回复。
    """
    messages = state['messages']
    persona = state.get('customer_persona', 'Budget Sensitive')
    # 默认为需求分析阶段，实际可扩展一个 'SalesManager' 节点来动态更新阶段
    stage = state.get('sales_stage', 'Needs Analysis') 
    
    # 1. 获取最后一条真人消息用于 RAG 检索
    last_human_msg = ""
    if messages and isinstance(messages[-1], HumanMessage):
        last_human_msg = messages[-1].content
    
    # 2. 调用 RAG 模块检索市场知识 (Module 1)
    # 只有当用户说了具体内容时才检索，否则使用空上下文
    context = rag.retrieve_info(last_human_msg) if last_human_msg else "No specific market context needed."
    
    # 3. 调用大脑生成回复 (Module 2)
    response: CustomerResponse = generate_customer_response(
        history=messages,
        persona=persona,
        stage=stage,
        context=context
    )
    
    # 4. 返回状态更新 (State Update)
    # 这里的字典会与原 State 进行 merge
    return {
        "messages": [AIMessage(content=response.spoken_response)],
        "latest_thought": response.thought_process,
        "status": response.status
    }

# ==========================================
# 4. 构建图 (Graph Construction)
# ==========================================
def compile_graph():
    """
    编译并返回 LangGraph 可运行对象
    """
    # 初始化图
    workflow = StateGraph(SalesState)
    
    # 添加节点
    workflow.add_node("customer", customer_node)
    
    # 设置入口点
    # 当用户输入 HumanMessage 后，图开始执行，直接进入 customer 节点处理
    workflow.set_entry_point("customer")
    
    # 添加条件边 (Conditional Edge)
    # 逻辑：customer 节点执行完后，检查 status 字段
    def should_continue(state: SalesState):
        status = state.get("status", "CONTINUE")
        if status in ["DEAL", "LEAVE"]:
            return END  # 结束对话
        return END      # 在 Demo 中，AI 回复一次后也暂停，等待用户下一次输入
        # 注意：如果是全自动模拟（AI vs AI），这里会指向另一个节点。
        # 在人机交互模式下，指向 END 意味着本次 Invoke 结束，把控制权交还给 Streamlit UI。

    workflow.add_edge("customer", END)
    
    # 编译图
    app = workflow.compile()
    return app