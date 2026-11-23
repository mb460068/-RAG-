# 该脚本用于：BM25 搜索，相似度 >= 0.85 时走 MySQL 问答系统。

# 导包
from rank_bm25 import BM25Okapi  # 导入 BM25 算法
import numpy as np  # 导入数值计算库
from mysql_qa.utils.preprocess import preprocess_text  # 导入文本预处理
from mysql_qa.db.mysql_client import MySQLClient  # 导入MySQL数据库操作
from mysql_qa.cache.redis_client import RedisClient  # 导入Redis缓存操作
from base import logger  # 导入日志


# 1. 定义 BM25 搜索类 -> 封装数据加载、模型初始化、相似度计算与答案检索。
class BM25Search:
    # 1.1 初始化方法，初始化：客户端实例，模型参数，加载配置。
    def __init__(self, redis_client, mysql_client):
        # 1. 日志实例。
        self.logger = logger
        # 2. Redis 和 MySQL实例。
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        # 3. 初始化BM25模型
        self.bm25 = None
        # 4. 初始化分词后的问题列表。
        self.questions = None
        # 5. 初始化原始问题列表 -> 用于匹配答案时的原始文本对照
        self.original_questions = None
        # 6. 调用数据加载的方法 -> 加载问题数据 并 初始化模型。
        self._load_data()

    # 1.2 数据加载方法 -> 优先从Redis缓存加载，缓存未命中则从MySQL加载并缓存。
    def _load_data(self):
        # 1. 定义Redis缓存键 -> 用于区分原始问题列表 和 分词后问题的缓存。
        original_key = 'qa_original_questions'  # 原始问题的缓存键
        tokenized_key = 'qa_tokenized_questions'  # 分词后问题缓存键

        # 2. 从Redis获取原始问题 -> 缓存读取极快，减少数据库访问。
        self.original_questions = self.redis_client.get_data(original_key)

        # 3. 从Redis获取分词后的问题。
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 4. 若Redis缓存中无数据(或者数据不完整), 则从MySQL加载并更新缓存。
        if not self.original_questions or not tokenized_questions:
            # 4.1 从MySQL中获取原始问题列表，格式：元组嵌套，例如：((问题1,), (问题2,),)
            self.original_questions = self.mysql_client.fetch_questions()
            # 4.2 若MySQL中也无问题数据，记录警告日志并返回。
            if not self.original_questions:
                self.logger.warning('未从MySQL中加载到任何问题数据')
                return

            # 4.3 走这里，说明原始问题肯定加载到了，对原始问题进行分词。
            tokenized_questions = [preprocess_text(question[0]) for question in self.original_questions]

            # 4.4 将原始问题列表 转换为 字符串列表，即：去除元组嵌套，并缓存到Redis中。
            self.redis_client.set_data(original_key, [question[0] for question in self.original_questions])

            # 4.5 将分词后的数据也缓存到Redis中。
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # 5. 保存分词后的问题列表，用于BM25模型初始化。
        self.questions = tokenized_questions

        # 6. 初始化BM25模型 -> 使用分词后的问题列表作为语料库。
        self.bm25 = BM25Okapi(self.questions)

        # 7. 记录INFO级别日志。提示模型初始化完成。
        self.logger.info('BM25模型初始化完成')

    # 1.3 softmax分数归一化方法 -> 将BM25分数转换为概率分布，便于阈值判断。
    def _softmax(self, scores):
        """
        函数功能: 对输入分数进行softmax归一化, 输出总和为1的概率分布
        :param scores: 用户问题 和 MySQL中各个问题的 bm25相似度分数
        :return: 归一化后的概率分布
        """
        # 1. 计算指数分数：每个分数 - 最大值，防止数值过大导致的溢出。
        exp_scores = np.exp(scores - np.max(scores))
        # 2. 返回归一化的结果，即：每个数占总和的百分比
        return exp_scores / exp_scores.sum()

    # 1.4 核心搜索方法 -> 处理查询，计算相似度，返回匹配答案 或者 提示无结果。
    def search(self, query, threshold=0.85):
        """
        根据输入查询检索醉相思的问题并返回对应的答案.
        :param query: 用户查询文本
        :param threshold: 相似度阈值 -> 超过此值认为匹配成功
        :return: 若匹配成功: (答案, False),  若未匹配或异常: (None, True)       True: 新查询, False: 旧查询
        """
        # ---------- 如下是从Redis中查询的动作 ----------
        # 1. 检查查询的有效性 -> 即：用户录入空值 或者 非字符串类型则视为无效。
        if not query or not isinstance(query, str):
            self.logger.error('无效查询: 查询为空或非字符串类型')
            return None, True

        # 2. 走这里，说明查询合法，先检查Redis缓存 -> 若查询结果已缓存，直接返回(提升效率)
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            # 走这里说明Redis匹配成功，返回缓存答案，即：标记 非新查询(旧查询，即：命中缓存)
            return cached_answer, False

        # ---------- 如下是从MySQL中查询的动作 ----------
        try:
            # 3. 对查询文本进行预处理，即：分词，且分词逻辑和 题库预处理逻辑一致。
            query_tokens = preprocess_text(query)
            # 4. 计算(用户)查询 和 所有问题的BM25相似度分数。
            scores = self.bm25.get_scores(query_tokens)
            # 5. 对分数进行归一化操作 -> 转换为概率分布，便于阈值判断。
            softmax_score = self._softmax(scores)
            # 6. 找到最高相似度对应的索引 和 分数。
            best_idx = softmax_score.argmax()
            best_score = softmax_score[best_idx]
            print(f'best_idx: {best_idx}, best_score: {best_score}')

            # 7. 若最高分不小于阈值，视为匹配成功。
            if best_score >= threshold:
                # 7.1 获取最高匹配度的原始问题 -> 用于从数据库查询答案。
                original_question = self.original_questions[best_idx]

                # 7.2 从数据库中查询答案。
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 7.3 将 <<查询-答案对>> 缓存到Redis中 -> 供下次使用。
                    # 键：用户问题；值：从 MySQL 中匹配到的原始问题对应的答案。
                    self.redis_client.set_data(f'answer:{query}', answer)
                    # 7.4 记录INFO日志，搜索成功及相似度。
                    self.logger.info(f'<MySQL>搜索成功, 最大相似度: {best_score:.3f}')
                    # 7.5 返回结果。
                    return answer, False

            # 8. 若未超过阈值，记录INFO级日志并返回无结果。
            self.logger.info(f'未找到可靠答案, 最大相似度: {best_score:.3f}, 低于阈值{threshold}')
            return None, True  # True表示(新查询), 即：需要(RAG)进一步处理。

        except Exception as e:
            # 9. 捕获搜索过程中所有的异常 -> 例如：模型未初始化，数据库连接失败...
            self.logger.error(f'搜索查询失败: {e}')
            return None, True  # True表示需要进一步处理。


# 2. 测试代码。
if __name__ == '__main__':
    # 1. 实例化Redis 和 MySQL的客户端。
    redis_client = RedisClient()
    mysql_client = MySQLClient()

    # 2. 实例化 BM25Search -> 自动加载数据并初始化模型。
    bm25_search = BM25Search(redis_client, mysql_client)

    # 3. 测试搜索功能：按问题检索最匹配的答案。
    result = bm25_search.search(query='可靠性优化设计一般包含哪三方面的内容？')
    print(f'result: {result}')
