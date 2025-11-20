import os
import shutil
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class GoldKnowledgeBase:
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = persist_dir
        self.embedding_function = OpenAIEmbeddings(model="BAAI/bge-m3", 
    base_url="https://api.siliconflow.cn/v1")
        self.vector_store = None
        self.retriever = None

    def initialize_knowledge_base(self, force_refresh: bool = False):
        if force_refresh and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_function
            )
        else:
            self._create_and_persist_data()

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 10}
        )

    def _create_and_persist_data(self):
        knowledge_data = [
            "今日国际金价为 580 元/克。",
            "本店（周大福模拟店）今日金价优惠后为 620 元/克（含工费）。",
            "隔壁竞品店（机器加工）价格为 590 元/克，但没有售后保障。",
            "古法金工艺特点：采用国家非物质文化遗产工艺，表面呈哑光质感，色泽温润。",
            "古法金优势：耐脏耐看，不易留指纹，硬度比普通黄金高，不易变形。",
            "普通黄金手镯（亮面）：容易产生划痕，且容易变形，适合预算极低的客户。",
            "本店售后服务：终身免费清洗、整形、编绳服务。",
            "关于工费贵的解释：古法金需要匠人手工敲打数万次，工费确实比机器做的贵，但更有收藏价值。"
        ]
        documents = [Document(page_content=text, metadata={"category": "sales_manual"}) for text in knowledge_data]
        self.vector_store = Chroma.from_documents(documents=documents, embedding=self.embedding_function, persist_directory=self.persist_dir)

    def retrieve_info(self, query: str) -> str:
        if not self.retriever:
            return "知识库未初始化"
        try:
            docs = self.retriever.invoke(query)
            return "\n".join([f"- {doc.page_content}" for doc in docs]) if docs else "暂无相关信息"
        except Exception:
            return "检索服务暂时不可用"
