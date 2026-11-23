# 该脚本用于：配置文件管理。

# 导包
import configparser  # 导入配置文件解析库，用于解析 ini格式 的配置文件。
import os  # 导入操作系统交互库，用于处理文件路径等操作。
from openai import OpenAI

# 1. 配置文件路径计算。
# 1.1 获取当前文件的绝对路径 -> 即：包含文件名的完整路径。
current_file_path = os.path.abspath(__file__)

# 1.2 获取当前文件的所在目录 -> 即：不包含文件名的路径。
current_dir_path = os.path.dirname(current_file_path)

# 1.3 获取项目根目录的绝对路径 -> 即：当前文件目录的上级目录。
# 目的：统一项目根目录路径，确保配置文件路径拼接的一致性。
project_root = os.path.dirname(current_dir_path)

# 1.4 拼接配置文件。
config_file_path = os.path.join(project_root, 'config.ini')


# 2. 配置解析类 -> 封装配置文件读取逻辑，提供统一的配置参数访问接口。
class Config():
    # 2.1 初始化配置，加载 config.ini 文件，并解析各服务的配置参数。
    def __init__(self, config_file=config_file_path):
        # ---------- 1.读取配置文件信息 ----------
        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 如果没有提供配置文件路径，则使用默认路径
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件
        self.config.read(config_file, encoding='utf-8')

        # ---------- 2.读取MySQL配置信息 ----------
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        # MySQL 密码
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        # MySQL 数据库名
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE',
                                        self.config.get('mysql', 'database', fallback='subject_kg'))

        # ---------- 3.读取Redis配置信息 ----------
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        # Redis 密码
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        # Redis 数据库编号
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # ---------- 4.读取Milvus配置信息 ----------
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='edurag'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name',
                                                                fallback='edurag_final'))

        # ---------- 5.读取 LLM(大模型) 配置信息 ----------
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='deepseek-chat')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key',
                                                                                fallback='sk-eaa4ba917f53420cbfdf9b841e2fecaf'))
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://api.deepseek.com/v1')

        # ---------- 6.读取分块相关信息 ----------
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=3)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # ---------- 7.读取应用配置信息 ----------
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["lx", "gx", "yx", "wx", "nx", "jw"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')
        # 扩展：创建Config类的实例 -> 用于访问配置参数


config = Config()

# 3. 程序主入口 -> 测试Config配置类的功能。
if __name__ == '__main__':
    # 3.1 手动指定配置文件的路径。
    config_file = '../../config.ini'

    # 3.2 实例化Config类，加载并解析配置文件。
    conf = Config()  # 不传参，用默认的路径。

    # 3.3 打印测试。
    print(conf.MYSQL_HOST, conf.MYSQL_USER, conf.MYSQL_PASSWORD, conf.MYSQL_DATABASE)  # MySQL 配置
    print(conf.REDIS_HOST, conf.REDIS_PORT, conf.REDIS_PASSWORD, conf.REDIS_DB)  # Redis 配置
    print(conf.LOG_FILE)  # 日志目录
    print(conf.MILVUS_HOST, conf.MILVUS_PORT, conf.MILVUS_DATABASE_NAME, conf.MILVUS_COLLECTION_NAME)  # Milvus 配置
    print(conf.LLM_MODEL, conf.DASHSCOPE_BASE_URL, conf.DASHSCOPE_API_KEY)  # LLM 配置
    print(conf.PARENT_CHUNK_SIZE, conf.CHILD_CHUNK_SIZE, conf.CHUNK_OVERLAP, conf.RETRIEVAL_K,
          conf.CANDIDATE_M)  # 父子块
    print(conf.VALID_SOURCES, conf.CUSTOMER_SERVICE_PHONE)  # 应用配置
