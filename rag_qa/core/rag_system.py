# 该脚本用于：高校课程智能助教的RAG核心逻辑 -> 实现RAG(检索增强生成)系统的核心逻辑，整合查询分类(意图识别), 策略选择，文档检索 和 答案生成。


# 1. 导包
import sys, os  # 导入路径处理和系统配置。
import time  # 导入 time 模块，用于计算时间

# 2. 配置项目路径 -> 确保程序能正确导入其它目录的模块。
# 2.1 获取当前文件的目录的绝对路径。
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2.2 获取当前目录的上级目录。
rag_qa_path = os.path.dirname(current_dir)
# 2.3 将rag_qa目录添加到Python的模块搜索路径。
sys.path.insert(0, rag_qa_path)
# 2.4 获取项目根目录。
project_root = os.path.dirname(rag_qa_path)
# 2.5 将项目根目录添加到Python的模块搜索路径。
sys.path.insert(0, project_root)

from rag_qa.core.prompts import RAGPrompts  # 导入RAG相关的提示模板 -> 定义大模型的输入和输出格式。
from rag_qa.core.query_classifier import QueryClassifier  # 导入查询分类器 -> 判断用户问题属于通用知识还是专业咨询，意图识别
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器 -> 专业咨询的情况下用哪种检索策略(例如:直接检索...)
from rag_qa.core.vector_store import VectorStore  # 导入向量数据库对象 -> 存储和检索文档向量。

from base.config import config  # 配置文件
from base.logger import logger  # 日志对象

# 加载项目配置
conf = config

# 3. 定义RAGSystem类 -> 封装RAG系统的完整流程，即：查询分类 -> 策略选择 -> 文档检索 -> 答案生成
class RAGSystem:
    # 3.1 初始化方法 -> 创建RAG系统所需的核心组件 -> 向量库，大模型，分类器...
    def __init__(self, vector_store, llm):
        """
        函数作用: 初始化RAG系统
        :param vector_store: 向量数据库对象 -> 用于存储和检索文档, 提供相似性搜索功能.
        :param llm: 大语言模型调用函数 -> 接收提示文本, 返回模型生成的回答.
        """
        # 1. 保存向量数据库对象。
        self.vector_store = vector_store
        # 2. 保存大模型调用函数 -> 用于生成答案，子查询，假设答案等...
        self.llm = llm
        # 3. 从 prompts.py 获取外部 RAG 提示模板。
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 分类标签沿用模板模型；'专业咨询'在此对应需要检索课程资料的问题。
        # 4. 初始化查询分类器 -> 用于判断用户查询是'通用知识'还是'专业咨询'
        # 4.1 拼接分类模型的路径。
        classifier_path = os.path.join(rag_qa_path, 'models', 'bert_query_classifier')
        # 4.2 创建查询分类器实例。
        self.query_classifier = QueryClassifier(classifier_path)
        # 5. 初始化策略选择器 -> 用于专业咨询的检索策略选择。
        self.strategy_selector = StrategySelector()

    # 3.1.1 工具方法：把LLM的返回值统一为完整文本 -> 兼容"一次性返回str"和"流式返回生成器"两种调用方式。
    @staticmethod
    def _to_text(llm_result):
        """
        函数作用: 统一大语言模型的返回结果类型.
        :param llm_result: LLM调用结果, 可能是str, 也可能是逐token输出内容的生成器.
        :return: 完整的答案文本(str).
        """
        if isinstance(llm_result, str):
            return llm_result
        # 生成器：拼接所有token, 得到完整答案。
        return "".join(llm_result)

    # 3.2 定义内部方法，使用假设文档进行检索（HyDE） -> 生成假设答案，用假设答案检索相关文档。
    def _retrieve_with_hyde(self, query, source_filter=None):
        """
        函数作用: 针对于抽象/开放性查询, 先生成假设性答案, 再用假设答案检索文档 -> 解决 抽象查询 直接检索关键词 匹配度低 的问题.
        :param query: 用户的原始查询文本(str).
        :param source_filter: 检索来源过滤条件(str或者None), 例如: education表示只检索与教育相关的文档,  None表示不过滤.
        :return: list[Document], 每个Document对象的page_content属性为文档文本, 用于后续生成答案的上下文.
        """
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        # 1. 获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        # 2. 调用大语言模型生成假设答案
        try:
            hypo_answer = self._to_text(self.llm(hyde_prompt_template.format(query=query))).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            # 3. 使用假设答案进行检索，并返回检索结果
            # 注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer,  # 检索关键词：生成的假设答案。
                k=conf.RETRIEVAL_K,  # 使用 K 而非 M
                source_filter=source_filter  # 应用来源过滤条件。
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    # 3.3 定义内部方法，使用子查询进行检索 -> 将复杂查询拆分为子查询，分别检索后合并结果。
    def _retrieve_with_subqueries(self, query, source_filter=None):
        """
        函数作用: 针对于多实体/多方面的复杂查询, 拆分为多个简单子查询, 分别检索后合并去重 -> 解决多维度查询检索不全面的问题.
        :param query: 用户的原始查询文本.
        :param source_filter: 检索来源过滤条件(str或者None), 例如: education表示只检索与教育相关的文档,  None表示不过滤.
        :return: list[Document]对象.
        """
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        # 获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            # 调用大语言模型生成子查询列表
            subqueries_text = self._to_text(self.llm(subquery_prompt_template.format(query=query))).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            # 初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            # 遍历每个子查询
            for sub_q in subqueries:
                # 使用子查询进行检索，并将结果添加到列表中
                # 这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K, source_filter=source_filter  # 应用来源过滤条件 # 使用 K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            # 对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            # 更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            # 返回去重后的文档，限制数量 (是否需要在此处限制？retrieve_and_merge 末尾会限制)
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    # 3.4 定义内部方法，使用回溯问题进行检索 -> 将复杂查询简化为基础问题后再检索。
    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        # 获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            # 调用大语言模型生成回溯问题
            simplified_query = self._to_text(self.llm(backtrack_prompt_template.format(query=query))).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            # 使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K, source_filter=source_filter  # 使用 K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    # 3.5 核心方法：根据检索策略检索文档，用 合并/筛选最终上下文对象。
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):  # 新增 strategy 参数
        """
        函数作用: 统一入口: 根据指定策略(或自动选择策略)调用对应检索方法, 筛选最终用于生成答案的上下文文档.
        :param query: 用户的原始查询文本 -> str, 传递给策略选择器 和 检索方法.
        :param source_filter: 检索来源过滤条件
        :param strategy: 指定检索策略, 可选: 直接检索, 回溯问题检索...
        :return: list[Document]
        """
        # 1. 如果未指定检索策略，自动选择检索策略。
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        # 2. 根据策略调用对应的检索方法 -> 获取候选文档列表。
        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter=source_filter)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter=source_filter)  # 返回的是唯一文档列表
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
            # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter=source_filter)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        # 3. 选择最终上下文文档，截取前conf.CANDIDATE_M个文档(控制上下文长度，避免LLM输入超限)
        logger.info(f"策略 {strategy} 检索到 {len(ranked_sub_chunks)} 个候选文档")
        final_context_docs = ranked_sub_chunks[: conf.CANDIDATE_M]  # 截取前 conf.CANDIDATE_M 个文档
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")

        # 4. 返回最终的上下文文档
        return final_context_docs

    # 3.6 定义方法，生成答案
    def generate_answer(self, query, source_filter=None, history=None):
        """
        函数作用: RAG系统对外核心接口 -> 接收用户查询, 自动完成'查询分类 -> 策略选择 -> 文档检索 -> 答案生成'全流程
        :param query: 用户录入的原始查询文本(字符串形式)
        :param source_filter: 检索来源过滤条件, 仅对'专业咨询'生效.
        :param history: 对话历史列表, 每项格式为 {'question': 问题, 'answer': 答案}.
        :return: 生成的最终答案文本 -> 字符串形式.
        """
        # 1. 记录查询开始时间：用户计算整个查询的处理耗时。
        start_time = time.time()
        logger.info(f"开始处理查询: {query}, 课程来源过滤: {source_filter}")

        # 1. 验证并处理对话历史，最多保留最近5轮。
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]
            for history_item in history:
                if not (
                        isinstance(history_item, dict)
                        and "question" in history_item
                        and "answer" in history_item
                ):
                    logger.warning(
                        f"无效的历史条目: {history_item}，忽略历史"
                    )
                    history = []
                    break

        # 2. 将历史记录整理为提示词文本。
        history_context = ""
        if history:
            history_context = "\n".join(
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in history
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        # 在当前提示词模板中加入独立的历史区域，保留原有课程资料上下文。
        prompt_template = self.rag_prompt.template.replace(
            "课程资料：",
            "对话历史：\n{history}\n课程资料：",
            1,
        )

        # 2. 调用查询分类器 -> 做：意图识别('通用知识' 或者 '专业咨询')
        query_category = self.query_classifier.predict_category(query)

        # 3. 若为'通用知识'类查询 -> 直接调用LLM生成答案，无需检索文档，例如：常识性问题。
        if query_category == "通用知识":
            logger.info("查询为通用知识, 直接调用LLM生成答案")

            # 3.1 构造LLM提示：通用知识无需上下文，仅传入：问题 和 回答依据。
            prompt_input = prompt_template.format(
                context="",  # 通用知识无上下文
                history=history_context,
                phone=conf.CUSTOMER_SERVICE_PHONE,
                question=query,  # 用户录入的要查询的问题
                answer_mode="本次未检索课程资料，仅作通用知识解释，不代表本课程的具体要求。"
            )

            try:
                # 3.2 调用LLM生成答案。
                answer = self.llm(prompt_input)
            except Exception as e:
                # 3.3 处理异常：LLM调用失败是，返回重试或咨询任课教师的提示。
                logger.error(f'直接调用LLM失败: {e}')
                answer = "抱歉，暂时无法回答这个问题，请稍后重试或咨询任课教师。"

            # 3.4 记录通用知识查询的处理结果 -> 耗时，查询内容。
            processing_time = time.time() - start_time
            logger.info(f"通用知识查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")

            # 3.5 返回最终的答案。
            return answer

        # 4. 若为'专业咨询'类查询，则执行完整RAG流程：策略选择 -> 文档检索 -> 结合上下文生成答案。
        logger.info("查询为专业咨询, 执行完整 RAG 流程")

        # 4.1 策略选择 -> 获取最合适的检索策略。
        strategy = self.strategy_selector.select_strategy(query)

        # 4.2 检索相关文档：调用retrieve_and_merge方法 -> 获取最终的文档列表。
        # 参1: 用户查询文本，参2: 检索来源过滤条件，参3: 检索策略。
        context_docs = self.retrieve_and_merge(query, source_filter=source_filter, strategy=strategy)

        # 4.3 构建上下文文本：将文档列表的page_content属性拼接为字符串(用换行分割，便于LLM阅读)
        if context_docs:
            # 走这里，说明检索到了文档，拼接即可。
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"构建上下文完成, 包含 {len(context_docs)} 个文档块")
        else:
            # 未找到课程依据时直接说明，避免生成没有依据的课程结论。
            logger.info("未检索到相关课程资料")
            return "课程知识库中暂未检索到相关资料，请补充课程名称、章节或问题细节，或向任课教师确认。"

        # 4.4 构建LLM提示：拼接：上下文文本，问题，回答依据。
        prompt_input = prompt_template.format(
            context=context,
            history=history_context,
            phone=conf.CUSTOMER_SERVICE_PHONE,
            question=query,
            answer_mode="依据检索到的课程资料回答；资料不足时明确说明，不确定的课程信息请向任课教师核实。"
        )

        try:
            # 4.5 调用LLM生成答案。
            answer = self.llm(prompt_input)
        except Exception as e:
            # 4.6 处理异常：LLM调用失败是，返回重试或咨询任课教师的提示。
            logger.error(f'直接调用LLM失败: {e}')
            answer = "抱歉，暂时无法生成课程解答，请稍后重试或咨询任课教师。"

        # 4.7 记录 专业咨询 查询的处理结果 -> 耗时，查询内容。
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")

        # 4.8 返回最终的答案。
        return answer


# 4. 测试代码 -> 需先准备分类模型、向量知识库和大模型配置。
# 运行前请确认 config.ini 中的模型与数据库配置已填写完整。
if __name__ == '__main__':
    # 1. 实例化向量数据库。
    vector_store = VectorStore()
    # 2. 使用模板的大语言模型调用函数。
    llm = StrategySelector().call_dashscope
    # 3. 创建高校课程智能助教的RAG核心对象。
    rag_system = RAGSystem(vector_store, llm)
    # 4. 提问；source_filter 应填写入库时已有的来源标签，None 表示不过滤。
    answer = rag_system.generate_answer('生物种群是什么')
    # 5. 打印课程解答。
    print(answer)
