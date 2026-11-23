# 该脚本用于：加载图片文件，通过 OCR 识别图片中的文字并转换为 Document。

from typing import Iterator
from rag_qa.edu_document_loaders.edu_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class OCRIMGLoader(BaseLoader):
    """基于 OCR 的图片文档加载器：把图片中的文字识别出来并封装成 Document。"""

    def __init__(self, img_path: str) -> None:
        """
        初始化图片加载器。

        参数:
            img_path: 待识别图片的文件路径。
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载图片内容：每次产出一个 Document，避免一次性占用过多内存。

        说明:
            lazy_load() 不接收参数，内部使用生成器逐个返回文档对象。
        """
        # 调用 OCR 将图片转换为文本
        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        """调用 OCR 引擎识别图片文字，并把多行结果拼接成一个字符串。"""
        resp = ""
        ocr = get_ocr()
        result, _ = ocr(self.img_path)
        if result:
            # result 中每项通常形如 [坐标，文本，置信度]，这里只取文本部分
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp


if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path='/0001.项目目录/others/test/人工智能就业课课程大纲.png')
    doc = img_loader.load()
    print(doc)