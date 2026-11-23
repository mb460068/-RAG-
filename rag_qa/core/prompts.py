# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate


# 1. 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # 1.1 定义 RAG 提示模板 -> 根据上下文生成答案，无上下文则用自身知识，无法回答时返回：客服信息。
    # 定义 RAG 提示模板
    @staticmethod
    def rag_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
                你是一个智能助手，帮助用户回答问题。请参考用户的助手的对话历史和上下文回答问题。
                如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
                如果答案来源于检索到的文档，请在回答中说明。

                对话历史: {history}
                上下文: {context}
                问题: {question}

                如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
                回答:
                """,
            # 定义输入变量
            input_variables=["context", "history", "question", "phone"],
        )

    # 1.2 定义假设问题生成的 Prompt 模板 -> 生成查询时的'假设性答案', 用于提升后续的检索精度。
    @staticmethod
    def hyde_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            # 定义输入变量
            input_variables=["query"],
        )

    # 1.3 定义子查询生成的 Prompt 模板 -> 将长/复杂查询拆分为多个简单子查询，便于分布检索。
    @staticmethod
    def subquery_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            # 定义输入变量
            input_variables=["query"],
        )

    # 1.4 定义回溯问题生成的 Prompt 模板 -> 将复杂/冗长查询简化为简短问题，提升检索关键词集中度。
    @staticmethod
    def backtracking_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            # 定义输入变量
            input_variables=["query"],
        )


# 2. 测试代码。
if __name__ == '__main__':
    # 测试1: 基础RAG回答模版 -> 直接检索。
    # 1. 创建 RAG 提示模板类的实例
    rag_prompt = RAGPrompts.rag_prompt()
    # 2. 测试RAG基础模板。
    result = rag_prompt.format(
        context="【参考文件：研究生院关于2024春季学期选课通知.docx，第2页】\n研究生选课分为初选和补退选两个阶段。初选时间为第1周至第2周。补退选时间为第3周。逾期系统将自动关闭，不再受理选课和退课申请。",
        question="我错过了初选，第3周还能退课吗？",
        history="用户：选课时间是什么时候？\nAI：分为初选（第1-2周）和补退选（第3周）阶段。",
        phone="12345678")
    # 3. 打印结果
    print(result)

    # 测试2: HyDE假设答案。
    # 1. 创建 HyDE 提示模板类的实例
    hyde_prompt = RAGPrompts.hyde_prompt()
    # 2. 测试HyDE模板。
    result = hyde_prompt.format(query="选课时间是什么时候？")
    # 3. 打印结果
    print(result)
