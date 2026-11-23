# 该脚本用于：Redis缓存技术。

# 导包
import redis
import json
import os, sys

# 路径说明：如果直接导入模块失败，可手动将项目根目录加入 sys.path。
# 导入自定义配置和日志模块。
from base import config, logger


# 1. 定义Redis客户端类 -> 封装Redis连接，数据存储，数据获取，答案查询等功能。
class RedisClient:
    # 1.1 初始化方法，获取Redis连接，初始化日志，处理连接异常。
    def __init__(self):
        # 1. 获取日志记录器。
        self.logger = logger

        # 2. 建立Redis连接。
        try:
            self.client = redis.StrictRedis(
                host=config.REDIS_HOST,  # Redis数据库主机地址
                port=config.REDIS_PORT,  # Redis数据库端口号
                password=config.REDIS_PASSWORD,  # Redis数据库密码
                db=config.REDIS_DB,  # Redis数据库编号
                decode_responses=True  # 自动将返回的字节数据转换为字符串(避免手动解码)
            )
        except redis.RedisError as e:
            self.logger.error(f'Redis连接异常: {e}')
            raise  # 声明抛出异常。

        # 3. 记录INFO日志
        self.logger.info('Redis连接成功')

    # 1.2 存储数据到Redis -> 将Python对象序列化为json字符串存储。
    def set_data(self, key, value):
        try:
            # 1. 存储数据。
            self.client.set(key, json.dumps(value, ensure_ascii=False))  # ensure_ascii: 保证中文等非ASCII字符正常显式。
            # 2. 记录INFO日志。
            self.logger.info(f'Redis存储数据成功: {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis存储数据失败: {e}')

    # 1.3 从Redis中获取数据 -> 反序列化为Python对象。
    def get_data(self, key):
        try:
            # 1. 获取数据。
            data = self.client.get(key)
            # 2. 处理结果 -> 反序列化为Python对象，再返回。
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis获取数据失败: {e}')
            return None  # 保持方法的返回值一致性。

    # 1.4 根据查询内容从Redis获取缓存的答案 -> 键格式固定为："answer:{query}"
    def get_answer(self, query):
        try:
            # 1. 构建键名，区分不同类型的缓存数据。
            answer = self.client.get(f"answer:{query}")
            # 2. 若存在缓存答案，记录INFO日志，并返回内容即可。
            if answer:
                self.logger.info(f'从Redis中获取答案成功: {query}')
                return answer
            # 3. 走到这里，说明缓存答案不存在，需要从数据库中查询，返回None
            return None
        except redis.RedisError as e:
            self.logger.error(f'从Redis中获取答案失败: {e}')
            return None


# 2. 测试代码
if __name__ == '__main__':
    # 1. 实例化RedisClient客户端。
    redCli = RedisClient()

    # 2. 获取Redis中所有的键。
    print(redCli.client.keys("*"))

    # 扩展：测试set_data()函数。
    redCli.set_data('user:1', {'name': '张三', 'age': 18})

    # 3. 测试get_data()方法：获取Redis中存储的答案。
    print(redCli.get_data('user:1'))

    # 4. 调用 get_answer() 获取 Redis 中缓存的答案
    print(redCli.get_answer('熵的解释'))
