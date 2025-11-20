# AI Gold Sales Agent

## 简介
这是一个基于 LangGraph + Streamlit 的销售模拟系统，包含：
1. **AI 客户模拟**：具备 3 种性格（预算敏感、独特设计、犹豫不决）。
2. **CoT 思维链**：展示 AI 决策的心理活动。
3. **RAG 知识库**：基于真实金价和工艺的市场数据检索。

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 Key：复制 `.env.example` 为 `.env` 并填入 OpenAI Key。
3. 运行系统：`streamlit run src/app.py`
