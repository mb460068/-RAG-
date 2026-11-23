# 该脚本用于：MySQL数据库的基础操作。


# 导包
import pymysql  # MySQL 数据库驱动
import pandas as pd  # 读取 CSV 文件，处理数据
import sys, os  # 处理路径

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录的绝对路径
module_dir = os.path.dirname(current_dir)  # 获取mysql_qa的目录
project_root = os.path.dirname(module_dir)  # 项目的根目录

sys.path.insert(0, project_root)  # 项目根目录添加至系统目录

from base import config, logger  # 导入全局配置(含MySQL账号密码等)和日志记录器


# 1. MySQL 客户端类 -> 封装连接、建表、插入、查询与关闭等功能。
class MySQLClient:
    # 1.1 初始化方法，建立数据库连接，创建游标，处理连接异常。
    def __init__(self):
        # 将全局logger赋值给属性，便于后续调用
        self.logger = logger

        try:
            # 建立mysql数据库连接
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,  # MySQL服务器主机地址
                user=config.MYSQL_USER,  # MySQL的用户名
                password=config.MYSQL_PASSWORD,  # MySQL登录密码
                database=config.MYSQL_DATABASE  # 要连接的数据库名
            )

            # 创建游标对象（执行sql语句，操作数据库）
            self.cursor = self.connection.cursor()

            # 日志记录
            self.logger.info("MySQL 连接成功")

        except pymysql.MySQLError as e:
            # 连接失败，抛出异常，终止后续操作，因为没有连接就无法进行任何数据库操作
            self.logger.error(f"MySQL 连接失败：{e}")
            raise

    # 1.2 创建数据表 -> 表名叫 jpkb，存储学科知识问答数据。
    def create_table(self):
        # 定义建表的SQL语句。(IF NOT EXISTS 防止重复建表报错)
        create_table_sql = """
                           CREATE TABLE IF NOT EXISTS jpkb
                           (
                               id
                               INT
                               AUTO_INCREMENT
                               PRIMARY
                               KEY,
                               subject_name
                               VARCHAR
                           (
                               20
                           ),
                               question VARCHAR
                           (
                               1000
                           ),
                               answer VARCHAR
                           (
                               1000
                           )) \
                           """

        try:
            # 执行sql语句 -> 创建表
            self.cursor.execute(create_table_sql)
            # 提交事务
            self.connection.commit()
            # 记录日志
            self.logger.info("表创建成功")
        except pymysql.MySQLError as e:
            # 创建失败
            self.logger.error(f'表创建失败：{e}')
            raise

    # 1.3 从 CSV 文件中批量获取数据。
    def insert_data(self, csv_path):
        try:
            # 读取 CSV 文件数据
            data = pd.read_csv(csv_path)

            # 遍历 CSV 每一行数据，执行插入操作
            for _, row in data.iterrows():
                # 定义插入数据的 SQL 语句，使用 %s 占位符避免 SQL 注入攻击
                insert_sql = 'INSERT INTO jpkb (subject_name,question,answer)VALUES (%s,%s,%s)'
                # 执行SQL语句（将 row 中的具体数据填入占位符并执行.）
                self.cursor.execute(insert_sql, (row['学科名称'], row['问题'], row['答案']))
            # 提交事务
            self.connection.commit()
            # 记录日志
            self.logger.info('MySQL数据插入成功。')

        except Exception as e:
            # 数据插入失败
            self.logger.error(f"数据插入失败：{e}")
            # 记得事务回滚，即：把数据还原到事务执行前的操作，防止插入一半的数据导致脏数据。
            self.connection.rollback()
            raise

    # 1.4 查询数据（获取表中所有问题）。
    def fetch_questions(self):
        try:
            # 执行sql查询，获取jpkb表中question列的所有数据
            self.cursor.execute("SELECT question FROM jpkb")
            # 从游标对象获取所有数据
            results = self.cursor.fetchall()
            # 记录日志
            self.logger.info("获取所有问题成功")
            # 返回结果
            return results
        except pymysql.MySQLError as e:
            # 获取数据失败
            self.logger.error(f"获取数据失败：{e}")
            return []  # 查询的所有问题，以列表形式返回

    # 1.5 根据指定问题，从 jpkb 表中获取答案。
    def fetch_answer(self, question):
        try:
            # 执行SQL查询
            self.cursor.execute("select answer from jpkb where question=%s", (question))

            # 获取查询数据
            result = self.cursor.fetchone()
            # 处理结果并返回
            return result[0] if result else None

        except pymysql.MySQLError as e:
            self.logger.error(f"获取数据失败：{e}")
            return None

    # 1.6 关闭连接，释放资源，避免连接泄露。
    def close(self):
        try:
            # 关闭MySQL连接和游标连接
            self.connection.close()
            # 记录日志
            self.logger.info("MySQL连接已关闭")

        except pymysql.MySQLError as e:
            # 连接关闭失败
            self.logger.error(f"关闭连接失败{e}")


# 2. 程序主入口
if __name__ == '__main__':
    # 1. 创建 MySQL 客户端。
    mysql_client = MySQLClient()
    print(f"当前连接的数据库是: {config.MYSQL_DATABASE}")
    # 2. 首次使用时，调用 create_table() 建表，再调用 insert_data() 导入数据。

    # 3. 查看表中所有的问题（用于测试验证）。
    results = mysql_client.fetch_questions()
    print(f'results: {results}, len: {len(results)}')

    # 4. 根据问题获取答案。
    result = mysql_client.fetch_answer('非诺贝特的安全性研究涵盖了哪些方面？')
    print(f'result: {result}')

    # 5. 关闭连接，释放资源（必须显式关闭，避免高并发时出现 Too many connections 错误）。
    mysql_client.close()
