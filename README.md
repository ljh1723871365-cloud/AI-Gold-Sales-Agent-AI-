👑 AI Gold Sales Agent (AI 黄金销售模拟系统)

基于 LangGraph + RAG + CoT (Chain of Thought) 构建的智能销售模拟 Agent。

📖 项目简介

本项目是一个面向销售场景的 AI 模拟系统。系统扮演一名具备复杂心理活动的“虚拟客户”，用户扮演“销售”，通过多轮对话考察销售技巧。

核心技术亮点：

LangGraph 状态机：管理“需求分析 -> 异议处理 -> 成交/离开”的完整销售生命周期。

CoT 思维链可视化：AI 在回复前会先进行内心独白（Inner Monologue），评估销售话术和价格。

RAG 检索增强：集成 ChromaDB 向量库，AI 基于真实的“今日金价”和“古法金工艺”数据进行决策，拒绝幻觉。

🏗️ 系统架构

graph TD
    User(真人销售) <--> UI[Streamlit 交互界面]
    UI <--> Agent[LangGraph 状态机]
    
    subgraph "AI Agent Brain"
        Persona[性格设定: 预算敏感/犹豫/独特]
        CoT[思维链推理引擎]
    end
    
    subgraph "Knowledge Base"
        RAG[(ChromaDB 向量库)]
        Data[金价行情/工艺手册]
    end
    
    Agent -- 读取状态 --> Persona
    Agent -- 检索知识 --> RAG
    CoT -- 决策生成 --> Agent


🚀 快速开始

1. 环境准备

确保已安装 Python 3.10 或更高版本。

# 克隆项目
git clone [https://github.com/你的用户名/gold_sales_agent.git](https://github.com/你的用户名/gold_sales_agent.git)
cd gold_sales_agent

# 安装依赖
pip install -r requirements.txt


2. 配置 API Key

复制配置文件模板：

cp .env.example .env


在 .env 中填入你的 API Key (支持 OpenAI / SiliconFlow 等):

OPENAI_API_KEY=sk-xxxxxx
OPENAI_API_BASE=[https://api.siliconflow.cn/v1](https://api.siliconflow.cn/v1)  # 如使用中转服务


3. 启动系统

streamlit run src/app.py


📂 目录结构

src/
├── modules/
│   ├── customer_brain.py  # AI 大脑 (CoT + Prompt)
│   ├── graph_builder.py   # LangGraph 状态机定义
│   └── rag_engine.py      # RAG 向量检索引擎
├── app.py                 # Streamlit 前端入口
└── ...


🧠 核心功能展示

多性格模拟：支持预算敏感型、追求独特型、犹豫不决型三种客户。

心理透视镜：实时展示 AI 的决策过程（如：“价格太贵，我要砍价”）。

动态成交系统：根据销售表现自动判定是否成交 (DEAL) 或 离开 (LEAVE)。

Created for Technical Interview Demo.
