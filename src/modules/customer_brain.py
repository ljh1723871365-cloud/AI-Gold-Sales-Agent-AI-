import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

# ==========================================
# 1. 定义数据结构 (Pydantic Model)
# ==========================================
class CustomerResponse(BaseModel):
    thought_process: str = Field(
        description="The internal reasoning process (Chain of Thought). CRITICAL: Analyze the salesperson's offer against market price and your persona before speaking."
    )
    spoken_response: str = Field(
        description="The actual words spoken to the salesperson. Keep it natural and consistent with your persona."
    )
    status: Literal["CONTINUE", "DEAL", "LEAVE"] = Field(
        description="Current conversation status. 'DEAL' if you buy, 'LEAVE' if you leave, 'CONTINUE' otherwise."
    )

# ==========================================
# 2. LLM 初始化
# ==========================================
def get_customer_llm():
    """
    获取 LLM 实例。自动读取 .env 中的配置。
    针对 SiliconFlow 进行了优化。
    """
    # 如果 .env 配置了 SiliconFlow，这里会自动使用
    # 建议使用 Qwen2.5-72B-Instruct 或 DeepSeek-V3，推理能力强
    return ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",  # 也可以换成 "deepseek-ai/DeepSeek-V3"
        temperature=0.6, # 温度适中，平衡创造性与指令遵循能力
        max_tokens=1024
    )

# ==========================================
# 3. 核心生成函数 (The Brain)
# ==========================================
def generate_customer_response(history, persona: str, stage: str, context: str) -> CustomerResponse:
    """
    核心逻辑：根据历史对话、人设、阶段和 RAG 知识生成回复。
    """
    llm = get_customer_llm()
    
    # 使用 Parser 确保输出格式稳定
    parser = PydanticOutputParser(pydantic_object=CustomerResponse)

    # --- System Prompt 设计 (核心考察点) ---
    # 采用了 "Role-Playing" + "Context-Injection" + "CoT-Enforcement"
    system_prompt_template = """
You are a virtual customer in a jewelry store simulating a real-world sales scenario.

=== 🎭 YOUR PERSONA: {persona} ===
- **Budget Sensitive**: You verify every price against market data. If > 600 RMB/g, you complain. You care about labor costs.
- **Unique Design**: You dislike common styles (like plain glossy). You want "Gu Fa Jin" (Ancient Method) or enamel. Price is secondary.
- **Indecisive**: You are easily swayed but hard to close. You always ask "What do you think?" or "Let me compare".

=== 📊 SALES STAGE: {stage} ===
(Needs Analysis -> Product Recommendation -> Objection Handling -> Closing)

=== 📚 MARKET KNOWLEDGE (RAG Context) ===
Use this data to fact-check the salesperson. Do NOT hallucinate prices.
{context}

=== 🧠 INSTRUCTIONS (Chain of Thought) ===
1. **CRITICAL THINKING**: Before generating a response, you MUST think internally:
   - Does the salesperson's offer match the market price provided in the context?
   - Does the product match my persona's taste?
   - Is the salesperson being pushy?
2. **DECISION**:
   - If they offer a good deal or answer your concern perfectly -> DEAL.
   - If they are rude, price is too high, or product is wrong -> LEAVE.
   - Otherwise -> CONTINUE.
3. **OUTPUT**:
   - Generate the JSON response strictly following the format below.

{format_instructions}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("placeholder", "{messages}"),
    ])

    # 将格式说明注入 Prompt
    partial_prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # 构建调用链
    chain = partial_prompt | llm | parser
    
    try:
        # 执行推理
        response = chain.invoke({
            "persona": persona,
            "stage": stage, # 实际项目中这个 Stage 可以由另一个 Chain 动态判断
            "context": context,
            "messages": history
        })
        return response

    except Exception as e:
        print(f"❌ [Customer Brain] Generation Error: {e}")
        # 兜底机制：防止 LLM偶尔抽风导致程序崩溃
        return CustomerResponse(
            thought_process=f"Error during reasoning: {str(e)}. I should ask for clarification.",
            spoken_response="不好意思，我刚才走神了，您能再说一遍吗？",
            status="CONTINUE"
        )