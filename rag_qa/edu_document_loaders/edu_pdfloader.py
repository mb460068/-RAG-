import cv2
import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
import numpy as np
from PIL import Image
from tqdm import tqdm
# 该脚本用于：基于 OCR 加载 PDF 文档，提取文本和图片中的文字。

from typing import Iterator
from rag_qa.edu_document_loaders.edu_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import CharacterTextSplitter
# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)


class OCRPDFLoader(BaseLoader):
    """基于 OCR 的 PDF 文档加载器：提取 PDF 中的文本和图片文字。"""

    def __init__(self, file_path: str) -> None:
        """
        初始化 PDF 加载器。

        参数:
            file_path: 待加载的 PDF 文件路径。
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载 PDF 内容：每次产出一个 Document，避免一次性占用过多内存。

        说明:
            lazy_load() 不接收参数，内部使用生成器逐个返回文档对象。
        """

        line = self.pdf2text()
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self):
        ocr = get_ocr()
        # 打开pdf文件
        doc = fitz.open(self.file_path)
        # 获取页数
        resp = ""
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")
        for i, page in enumerate(doc):
            b_unit.set_description("OCRPDFLoader context page index: {}".format(i))
            b_unit.refresh()
            # 提取文本：默认使用 "text" 模式提取文本。
            text = page.get_text("text")
            resp += text + "\n"
            # 获取图片：获得所有显示的图像的元信息列表。
            # 它适用于所有文档类型，不仅限于 PDF。
            img_list = page.get_image_info(xrefs=True)
            for img in img_list:
                # xref一种编号，指向该图像对象在PDF文件中的位置，程序可以通过这个编号快速定位和提取图像数据。
                if xref := img.get("xref"):
                    # 图像在页面上的位置和尺寸。
                    bbox = img["bbox"]
                    # 检查图片尺寸是否超过设定的阈值
                    if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                            or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                        continue
                    pix = fitz.Pixmap(doc, xref)
                    if int(page.rotation) != 0:  # 如果Page有旋转角度，则旋转图片
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        tmp_img = Image.fromarray(img_array)
                        ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                        rot_img = self.rotate_img(img=ori_img, angle=360 - page.rotation)
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                    # result：包含了图像中检测到的所有文本框的位置、文本内容和置信度信息。
                    # _：它是一个包含了时间数据的列表，可以用于优化模型运行速度。
                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)
            # 更新进度
            b_unit.update(1)
        return resp

    def rotate_img(self, img, angle):
        """
        按指定角度旋转图片。

        参数:
            img: 待旋转的图片数组。
            angle: 旋转角度。
        返回:
            旋转后的图片数组。
        """

        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        # 获取旋转矩阵
        # 参数1为旋转中心点;
        # 参数2为旋转角度，正值-逆时针旋转；负值-顺时针旋转
        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算图像新边界
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img

if __name__ == '__main__':
    # 测试：加载本地 PDF 并打印提取结果
    from base.config import Config
    pdf_loader = OCRPDFLoader(file_path=f"{Config().DATA_DIR}/ai_data/LLM基础知识.pdf")
    doc = pdf_loader.load()

    print(type(doc))
    print(doc)
