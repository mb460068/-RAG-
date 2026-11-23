# 该脚本用于：基于达摩院语义分割模型的中文文本切分器。

import warnings
warnings.filterwarnings("ignore")
from langchain_text_splitters import CharacterTextSplitter
import re
from typing import List
from modelscope.pipelines import pipeline


class AliTextSplitter(CharacterTextSplitter):
    """基于达摩院语义分割模型的文本切分器，适合按语义边界切分长文档。"""

    def __init__(self, pdf: bool = False, **kwargs):
        """
        初始化语义切分器。

        参数:
            pdf: 是否为 PDF 文本；为 True 时会先清理多余换行和空白字符。
        """
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        """调用语义分割模型，把一段长文本切分为若干句子/语义片段。"""
        # 使用达摩院开源的 nlp_bert_document-segmentation_chinese-base 做语义切分，论文见 https://arxiv.org/abs/2107.09278
        # 如需使用该模型，需要安装 modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 模型默认放在 CPU 上运行，以兼容低配置环境；如有需要可把 device 改为自己的显卡编号
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        from base.config import Config
        p = pipeline(
            task="document-segmentation",
            model=f'{Config().MODELS_DIR}/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
if __name__ == '__main__':
    model_split = AliTextSplitter()
    result = model_split.split_text(text='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')
    print(result)