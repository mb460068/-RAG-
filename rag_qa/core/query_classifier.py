# 该脚本用于：基于BERT模型实现查询分类器，将用户查询分为'通用知识'和'专业咨询'两大类。
# 核心流程：加载预训练BERT模型 -> 数据预处理 -> 模型训练 -> 模型评估 -> 预测查询类别

# 导包以及准备工作
# 禁用oneDNN优化
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 忽略所有警告
import warnings

warnings.filterwarnings('ignore')

import json  # 导入 JSON 模块，用于解析和读取 json 格式的训练数据集
import sys  # 导入系统模块，用于修改 Python 的模块搜索路径 (sys.path)

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
rag_qa_path = os.path.dirname(current_dir)  # rag_qa文件夹路径
project_root = os.path.dirname(rag_qa_path)  # 项目根目录路径
sys.path.insert(0, project_root)  # 项目目录添加到系统路径

import torch  # pytorch框架
import numpy as np  # 数值计算
from base.logger import logger  # 日志
from transformers import BertTokenizer, BertForSequenceClassification  # bert分词器和序列分类模型
from transformers import Trainer, TrainingArguments  # 训练器和参数配置设置
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.metrics import classification_report, confusion_matrix  # 评估工具


# 定义QueryClassifier类：封装BERT查询分类的完整流程 -> 模型加载，训练，评估，预测
class QueryClassifier:
    def __init__(self, model_path='../models/bert_query_classifier'):
        """
        初始化查询分类器
        :param model_path:模型保存/加载的路径
        """
        # 模型存储/加载路径
        self.model_path = model_path

        # 加载BERT分词器
        # 拼接预训练BERT模型的本地路径
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # 模型初始化（后续通过load_model()决定是否加载已有模型还是新建模型）
        self.model = None
        # 选择运行设备（cpu/gpu）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 日志记录
        logger.info(f"使用设备:{self.device}")
        # 定义标签映射字典（后续将通用知识转换为0，专业知识转换为1）
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 初始化时自动调用load_model()，保证实例化后模型随时可用
        self.load_model()

    # 实现加载模型，从指定路径加载已经训练的模型，若不存在则初始化新模型
    def load_model(self):
        """
        加载模型：优先查询self.model_path，不存在则初始化基于bert-base-chinese的新模型
        :return:
        """
        # 判断模型是否已训练并保存
        if os.path.exists(self.model_path):
            # 模型存在，从指定路径加载模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 加载模型到指定运行设备(cpu/gpu)
            self.model.to(self.device)
            # 日志记录
            logger.info(f"已加载训练后的模型：{self.model_path}")
        else:
            # 模型不存在，初始化新模型
            self.model = BertForSequenceClassification.from_pretrained(
                os.path.join(rag_qa_path, "models", "bert-base-chinese"), num_labels=2)
            # 加载模型到指定运行设备(cpu/gpu)
            self.model.to(self.device)
            # 日志记录
            logger.info(f"模型不存在，已初始化新模型")

    # 模型保存
    def save_model(self):
        # 模型参数和配置保存到指定文件夹（self.model_path）
        self.model.save_pretrained(self.model_path)
        # 分词器词表和配置文件也保存到同一文件夹，方便一起加载
        self.tokenizer.save_pretrained(self.model_path)
        # 日志记录
        logger.info(f"已保存模型至：{self.model_path}")

    # 数据预处理（将文本和标签转换为BERT模型所需要的格式）
    def preprocess_data(self, texts, labels):
        """
        对文本进行分词，截断，填充，将标签转换为数字
        :param texts: 待处理的文本列表
        :param labels: 文本的对应的标签
        :return: 处理后的编码和标签
        """
        # 文本编码，使用分词器将字符串转换为模型认识的输入格式
        encodings = self.tokenizer(
            texts,  # 待处理的文本
            truncation=True,  # 超过最大长度时截断
            padding='max_length',  # 不足最大长度是填充
            max_length=128,  # 最长长度128
            return_tensors='pt'  # 返回Torch张量
        )

        # 标签转换，文本标签转换为数字标签
        labels = [self.label_map[label] for label in labels]

        # 返回编码和标签
        return encodings, labels

    # 创建数据集方法类：自定义Dataset类，封装编码和标签
    def create_dataset(self, encodings, labels):
        # 函数内部定义一个Dataset类，继承PyTorch中的torch.utils.data.Dataset
        class Dataset(torch.utils.data.Dataset):
            # 初始化，接收特征和标签
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings  # 保存分词后的特征字典
                self.labels = labels  # 保存数字标签列表【1，0，1】

            # 返回数据集的总样本数
            def __len__(self):
                return len(self.labels)

            # 通过索引idx获取单个样本
            def __getitem__(self, idx):
                # 从 encodings 中提取第 idx 个样本的所有特征 (如 input_ids[idx], attention_mask[idx])
                item = {key: value[idx] for key, value in self.encodings.items()}
                # 去除对应的标签，并封装成PyTorch张量
                item["labels"] = torch.tensor(self.labels[idx])
                # 返回指定数据
                return item

        # 实例化内部类，并返回Dataset对象
        return Dataset(encodings, labels)

    # 模型训练：加载数据集，预处理，配置训练参数并训练模型
    def train_model(self, data_file="model_train.json"):
        """
        训练BERT分类模型, 区分查询类型为 <<通用知识>> 和 <<专业咨询>>
        :param data_file:数据集路径
        :return:
        """
        # 检查数据集文件是否存在，不存在则主动打断程序的继续执行，并大声报警，把错误信息抛给调用它的上层代码或控制台
        if not os.path.exists(data_file):
            logger.error(f"数据集文件{data_file}不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        # 数据集加载
        with open(data_file, "r", encoding='utf-8') as f:
            # 按行读取(拿到的字符串列表)，并解析为JSON对象，存入列表中
            data = [json.loads(value) for value in f.readlines()]

        # 从解析后的字典列表提取文本和标签
        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        # 划分训练集和测试集(20％用于验证，80％用于训练；随机种子固定为34)
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2,
                                                                            random_state=34)
        # 调用预处理函数，对文本分词并进行标签数字化
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        # 调用创建数据集函数，将特征和标签封装为PyTorch的Dataset对象
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir="./bert_results",  # 模型和检查点保存到临时目录
            num_train_epochs=5,  # 训练轮数
            per_device_train_batch_size=8,  # 训练批次大小
            per_device_eval_batch_size=8,  # 评估批次大小
            warmup_steps=20,  # 预热步数（20步学习率从0线性增加到设定值，防止初始震荡）
            weight_decay=0.01,  # 权重衰减系数（L2正则化），预防过拟合
            logging_dir="./bert_logs",  # 日志保存目录
            logging_steps=10,  # 每多少步记录一次训练日志
            evaluation_strategy="epoch",  # 评估策略（一轮以保存）
            save_strategy="epoch",  # 保存策略（一轮以保存）
            load_best_model_at_end=True,  # 训练结束后，自动加载评估指标最好的检查点
            save_total_limit=1,  # 保留检查点的数目（防止磁盘空间被占满）
            metric_for_best_model="eval_loss",  # 评估指标（验证集的loss最小）
            fp16=False  # 是否使用16位浮点数混合精度训练（这里表示使用的是FP32，更稳定）
        )

        # 初始化训练器
        trainer = Trainer(
            model=self.model,  # 传入要训练的模型
            args=training_args,  # 传入定义的训练参数
            train_dataset=train_dataset,  # 传入训练数据集
            eval_dataset=val_dataset,  # 传入验证数据集
            compute_metrics=self.computer_metrics  # 传入自定义评估指标计算函数
        )

        # 模型训练
        logger.info("开始训练BERT模型")
        trainer.train()

        # 训练完成后保存模型
        self.save_model()

        # 进行模型评估
        self.evaluate_model(val_texts, val_labels)

    # 模型评估指标
    def computer_metrics(self, eval_pred):
        """
        计算分类任务的评估指标 -> 准确率
        :param eval_pred: 包含模型输出 logits 和 真实标签的元组
        :return: 准确率字典
        """
        # 从 Trainer 传入的 eval_pred 中解包出模型原始输出 (logits) 和真实标签
        logits, labels = eval_pred
        # 在 logits 的最后一个维度上取最大值的索引，作为模型的预测类别 (0 或 1)
        predictions = np.argmax(logits, axis=-1)
        # 计算准确率
        accuracy = (predictions == labels).mean()
        # 返回字典格式（trainer会自动记录到日志中）
        return {"accuracy": accuracy}

    # 模型评估方法：生成分类报告和混淆矩阵，全面分析模型的性能
    def evaluate_model(self, texts, labels):
        """
        在给定文本和标签上评估模型，输出分类报告和混淆矩阵
        :param texts:待评估的文本列表
        :param labels:文本对应的真实标签
        :return:
        """
        # 对文本进行编码
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        # 创建评估用的Dataset对象
        dataset = self.create_dataset(encodings, labels)

        # 初始化一个不带训练参数的Trainer，仅用于执行预测
        trainer = Trainer(model=self.model)
        # 执行预测（返回包含 predictions (logits) 和 label_ids 的 PredictionOutput 对象）
        predictions = trainer.predict(dataset)

        # 解析预测结果（取 logits 最大值对应的索引作为预测标签）
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels  # 真实标签

        # 使用sklearn输出详细的评估报告到日志
        logger.info("分类报告：")
        # 输出每个类别的Precision（精确率）、recall（召回率）、F1-score 等
        logger.info(classification_report(true_labels, pred_labels, target_names=["通用知识", "专业咨询"]))
        logger.info("混淆矩阵：")
        # 输出混淆矩阵（直观查看哪类被分错成哪类）
        logger.info(confusion_matrix(true_labels, pred_labels))

    # 预测类别方法：对单个查询文本进行分类预测
    def predict_category(self, query):
        """
        预测单个查询的类别 -> 通用知识或专业咨询
        :param query: 待分类的查询文本
        :return: 分类结果（文本标签）
        """
        # 检查模型是否已经加载，防止空指针异常
        if self.model is None:
            logger.error("模型未训练/未加载")
            return "通用知识"  # 兜底返回

        # 对用户输入的单个查询文本进行编码
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        # 将编码后的数据移动到模型所在设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # 模型推理
        with torch.no_grad():
            # 将数据输入模型，获取前向传播的输出
            outputs = self.model(**encoding)
            # 获取 logits 在维度 1 上的最大索引，并使用 .item() 转换为 Python 的整数标量
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # 根据预测的数字结果返回文本标签（1 对应专业咨询，0 对应通用知识）
        return "专业咨询" if prediction == 1 else "通用知识"


# 运行测试
if __name__ == '__main__':
    # 1. 创建QueryClassifier对象 -> 即：实例化查询分类器
    query_classify = QueryClassifier()

    # 2. 训练模型（重新训练时使用）。
    data_file = "../classify_data/model_train.json"
    query_classify.train_model(data_file)

    # 3. 预测查询类别。
    result = query_classify.predict_category("生物学论文要怎么写")
    print(result)
