# 用于向量存储和检索。

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rag_qa.core.document_processor import *
import hashlib
import sys

# 配置项目目录
# 获取当前文件所在目录的绝对路径。
local_path = os.path.abspath(os.path.dirname(__file__))
# 获取当前目录的上级目录。
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
# 获取项目根目录的绝对路径。
project_root = os.path.dirname(rag_qa_path)
# 将项目根目录加入系统路径，确保跨目录导入自定义模块
sys.path.insert(0, project_root)

from base.config import config
from base.logger import logger


# 定义向量存储类
class VectorStore:
    # 配置向量库连接
    def __init__(
            self,
            collection_name=config.MILVUS_COLLECTION_NAME,  # Milvus集合名
            host=config.MILVUS_HOST,  # Milvus主机地址
            port=config.MILVUS_PORT,  # Milvus端口
            database=config.MILVUS_DATABASE_NAME  # Milvus数据库名
    ):
        # 存储 Milvus核心配置参数
        self.collection_name = collection_name  # 集合名
        self.host = host  # Milvus主机地址
        self.port = port  # Milvus默认端口号
        self.database = database  # Milvus的数据库名

        # 初始化日志实例
        self.logger = logger
        # 选择模型运行设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 记录日志
        self.logger.info(f'使用设置: {self.device}')
        # 初始化 BGE-Reranker重排序模型
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        # 加载模型
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # 拼接 BGE-M3嵌入模型的路径
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        # 加载模型
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name=m3_path,  # 模型本地路径
            use_fp16=(self.device == 'cuda'),  # GPU时启用半精度计算，cpu禁用，降低内存占用
            device=self.device  # 模型运行设备
        )

        # 获取 BGE-M3稠密向量维度 -> 固定输出 1024维。
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus客户端：建立与向量库的连接。
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用内部函数：创建新集合
        self._create_or_load_collection()

    # 不存在就创建新集合 或 加载已有集合
    def _create_or_load_collection(self):
        # 检查集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 走这里，说明集合不存在，需要创建集合 允许 $meta字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加论文题目字段，VARCHAR 类型，最大长度 255
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=255)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}  # 构建索引时，丢弃20%的低权重值(减少存储，噪声，精度影响较小)
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")

        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 将文档转换为向量，并插入Milvus中
    def add_documents(self, documents):
        # 提取所有文档的内容列表，从每个 document对象中获取 文档文本
        texts = [doc.page_content for doc in documents]
        # 使用 BGE-M3生成文档的向量表示 输出 稠密向量 和 稀疏向量
        embeddings = self.embedding_function(texts)
        # 初始化空列表，存储待插入Milvus的数据
        data = []

        # 遍历每个文档
        for i, doc in enumerate(documents):
            # 生成文档唯一ID
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # 处理稀疏向量
            sparse_vector = {}
            # 获取第 i个文档的稀疏向量行
            row = embeddings['sparse'][[i]]
            # 获取稀疏向量的非零值索引(indices) 和 对应权重(values)
            indices, values = row.indices, row.data
            # 组装稀疏向量字段，将索引和权重配对,
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value

            # 组装单条插入数据
            data.append({
                "id": text_hash,  # 唯一ID(MD5哈希)
                "text": doc.page_content,  # 文档内容
                "dense_vector": embeddings['dense'][i],  # 稠密向量
                "sparse_vector": sparse_vector,  # 稀疏向量
                "parent_id": doc.metadata["parent_id"],  # 父文档ID
                "parent_content": doc.metadata["parent_content"],  # 父文档内容
                "source": doc.metadata.get("source", "unknown"),  # 学科类
                "title": doc.metadata.get("title", "未知文献"),  # 论文题目
                "timestamp": doc.metadata.get("timestamp", "unknown")  # 时间戳
            })

        # 插入数据到 Milvus，仅当 data非空时执行。
        if data:
            # 有就更新，没有就插入
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f'已更新或插入 {len(data)} 个文档')

    # 混合检索(稠密 + 稀疏) + 重排序，返回精准父文档
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        # 生成查询的嵌入向量
        query_embeddings = self.embedding_function([query])
        # 提取查询的稠密向量：第 0个元素。
        dense_query_vector = query_embeddings["dense"][0]
        # 处理查询的稀疏向量 -> 转换为 Milvus支持的字典格式。
        sparse_query_vector = {}
        # 获取查询稀疏向量的第 0行 -> 仅 1个查询，所以取第 0行。
        row = query_embeddings["sparse"][[0]]
        # 获取非零值索引和权重。
        indices, values = row.indices, row.data
        # 组装稀疏向量字典。
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # 按学科过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # 定义稠密向量的检索参数。
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],  # 查询向量，列表格式，即使是单个。
            anns_field="dense_vector",  # (检索的)稠密向量字段名，即Schema中定义的字段名。
            param={"metric_type": "IP", "params": {"nprobe": 10}},  # 检索参数，度量类型(IP: 内积相似度), nprobe(检索时访问的聚类数，平衡速度与精度)
            limit=k,  # 返回的Top-K结果。
            expr=filter_expr  # 检索过滤表达式，按学科过滤。
        )

        # 定义稀疏向量的检索参数。
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],  # 查询向量，列表格式，即使是单个。
            anns_field="sparse_vector",  # (检索的)稀疏向量字段名，即Schema中定义的字段名。
            param={"metric_type": "IP", "params": {}},  # 检索参数，度量类型(IP: 内积相似度)
            limit=k,  # 返回的Top-K结果。
            expr=filter_expr  # 检索过滤表达式，按学科过滤。
        )

        # 创建加权排序器 融合稠密 和 稀疏检索的结果
        ranker = WeightedRanker(1.0, 0.7)

        # 执行混合检索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,  # 目标集合
            reqs=[dense_request, sparse_request],  # 检索请求列表, [稠密，稀疏]
            ranker=ranker,  # 加权排序器
            limit=k,  # 返回的Top-K结果。
            output_fields=["text", "parent_id", "parent_content", "title", "source", "timestamp"]  # 返回的字段
        )[0]  # 取第0个结果

        # 将检索结果转换为 LangChain Document对象
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]

        # 从子块中提取去重的父文档，避免同一个父块的多个子块重复返回。
        parent_docs = self._get_unique_parent_docs(sub_chunks)

        # 重排序逻辑：父文档数量 < 3 时，跳过重排序
        if len(parent_docs) < 3:
            return parent_docs[:conf.CANDIDATE_M]

        # 当父文档数量 >=3时，进行重排序
        if parent_docs:
            # 构建"查询-文档"配对列表 -> 重排模型需要按照这个格式输入，每个配对信息为 [query, doc_content]
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 计算相关性得分 -> BGE-Reranker预测每个配对的得分，得分越高，相关性越强。
            scores = self.reranker.predict(pairs)
            print(f'scores: {scores}')

            # 按得分降序排序，将得分与父文档配对，从高到低排序。
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), key=lambda pair: pair[0],
                                                           reverse=True)]  # 更完善一点，能排除分数一致的情况。
        else:
            # 走这里，说明父文档为空 -> 无检索结果，则返回空列表即可。
            ranked_parent_docs = []

        # 返回重排序后的Top-M父文档，取前M个结果。
        return ranked_parent_docs[:conf.CANDIDATE_M]

    # 从子块列表中提取去重的父文档。
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # 将 Milvus检索结果(hit) 转换为 LangChain Document对象。
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "title": hit.get("title"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )


# 4. 程序入口 -> 文档入库 / 混合检索
if __name__ == '__main__':
    # 1. 实例化 VectorStore 对象。
    vector_store = VectorStore()

    # 2. 测试混合检索 + 重排序（可按资料范围过滤）。
    query = "学籍管理办法？"

    results = vector_store.hybrid_search_with_rerank(query, source_filter="jw")

    # 3. 打印检索结果。
    print(f'results: {results}')  # Document 对象列表。
    print(f'results: {len(results)}')  # 打印结果数量
