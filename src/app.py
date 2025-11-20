import streamlit as st
import os
import sys
import time
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# 加载环境变量 (API Key 等)
load_dotenv()

# 确保 Python 能找到模块路径 (解决 ModuleNotFoundError)
sys.path.append(os.getcwd()) 

# 导入图构建器
from src.modules.graph_builder import compile_graph

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="AI 黄金销售模拟系统",
    page_icon="👑",
    layout="wide", # 使用宽屏模式以便左右分栏
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 状态初始化 (Session State)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = compile_graph() # 初始化 LangGraph
if "latest_thought" not in st.session_state:
    st.session_state.latest_thought = "等待对话开始... (此处将实时显示 AI 的思维链)"
if "current_status" not in st.session_state:
    st.session_state.current_status = "CONTINUE"
if "sales_stage" not in st.session_state:
    st.session_state.sales_stage = "Needs Analysis" # 初始阶段

# ==========================================
# 3. Sidebar: 控制面板
# ==========================================
st.sidebar.title("🛠️ 模拟配置")
st.sidebar.markdown("---")

# 性格选择
persona_display = st.sidebar.selectbox(
    "🎭 选择 AI 客户性格",
    (
        "Budget Sensitive (预算敏感型)", 
        "Unique Design (追求独特型)", 
        "Indecisive (犹豫不决型)"
    ),
    help="不同的性格会加载不同的 System Prompt 和 RAG 侧重点"
)

# 映射显示名称到内部 Key
persona_map = {
    "Budget Sensitive (预算敏感型)": "Budget Sensitive",
    "Unique Design (追求独特型)": "Unique Design", 
    "Indecisive (犹豫不决型)": "Indecisive"
}
selected_persona = persona_map[persona_display]

# 销售阶段显示 (可手动调整用于测试，或者由 AI 自动判断)
stage_display = st.sidebar.selectbox(
    "📊 当前销售阶段 (模拟)",
    ("Needs Analysis", "Product Recommendation", "Objection Handling", "Closing"),
    index=0
)
st.session_state.sales_stage = stage_display

st.sidebar.markdown("---")

# 重置按钮
if st.sidebar.button("🔄 重置对话 / 开始新模拟", type="primary"):
    st.session_state.messages = []
    # 重新编译图以确保状态清空
    st.session_state.graph = compile_graph()
    st.session_state.latest_thought = "等待对话开始..."
    st.session_state.current_status = "CONTINUE"
    st.rerun()

# 面试官提示
st.sidebar.info(
    "💡 **演示指南**：\n"
    "1. 选择一个性格。\n"
    "2. 在聊天框输入销售话术。\n"
    "3. 观察右侧的 CoT 思维链。\n"
    "4. 尝试触发 RAG (如询问金价对比)。"
)

# ==========================================
# 4. 主界面布局 (左右分栏)
# ==========================================
st.title("👑 AI Gold Sales Agent Simulation")
st.caption("Human (Salesperson) vs AI (Customer) | Architecture: LangGraph + RAG + CoT")

col_chat, col_thought = st.columns([0.65, 0.35], gap="large")

# --- 右侧：AI 心理活动透视镜 (CoT Visualization) ---
with col_thought:
    st.subheader("🧠 AI 心理活动透视")
    st.caption("面试加分项：实时展示 Chain-of-Thought 推理过程")
    
    # 使用容器美化展示
    with st.container(border=True):
        st.markdown(f"**当前人设**: `{selected_persona}`")
        st.markdown(f"**当前阶段**: `{st.session_state.sales_stage}`")
        st.divider()
        
        # 动态显示思维链
        st.markdown("#### 💭 Inner Monologue:")
        if st.session_state.latest_thought:
            st.info(st.session_state.latest_thought)
        else:
            st.text("Thinking...")

# --- 左侧：聊天窗口 ---
with col_chat:
    st.subheader("💬 销售对话现场")

    # 1. 渲染历史消息
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)

    # 2. 处理特殊状态 (成交/离开)
    if st.session_state.current_status == "DEAL":
        st.balloons()
        st.success("🎉 **成交达成！** 客户已决定购买。请点击侧边栏重置。")
    elif st.session_state.current_status == "LEAVE":
        st.error("🚪 **客户离开了。** 销售失败。请点击侧边栏重置。")
    
    # 3. 输入框 (仅在 CONTINUE 状态下显示)
    if st.session_state.current_status == "CONTINUE":
        if prompt := st.chat_input("作为销售，请输入话术 (例如：这款手镯做工非常精细...)"):
            # A. 显示用户消息
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(prompt)

            # B. AI 思考中 (Spinner)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI 正在结合行情进行思考 (RAG + CoT)..."):
                    try:
                        # 准备输入数据
                        inputs = {
                            "messages": st.session_state.messages,
                            "customer_persona": selected_persona,
                            "sales_stage": st.session_state.sales_stage,
                            "status": "CONTINUE"
                        }
                        
                        # 调用 LangGraph
                        # stream=False 直接获取结果，stream=True 可做打字机效果
                        result = st.session_state.graph.invoke(inputs)
                        
                        # 解析结果
                        ai_content = result["messages"][-1].content
                        thought_process = result.get("latest_thought", "No thought captured.")
                        status = result.get("status", "CONTINUE")
                        
                        # 更新 Session State
                        st.session_state.messages.append(AIMessage(content=ai_content))
                        st.session_state.latest_thought = thought_process
                        st.session_state.current_status = status
                        
                        # 强制刷新以更新右侧 CoT 面板
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"系统错误: {str(e)}")
                        st.info("提示：请检查 API Key 是否正确，或网络连接是否正常。")