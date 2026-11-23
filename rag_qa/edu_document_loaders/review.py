# 该脚本用于：演示将 PPT 中的文本、表格与图片内容提取为纯文本。

from tqdm import tqdm
import time
from PIL import Image
from io import BytesIO
from edu_ocr import *
import numpy as np
ocr = get_ocr()
from pptx import Presentation


def ppt2text(filepath):
    # 打开指定路径的 PowerPoint 文件
    prs = Presentation(filepath)
    resp = ''
    # 逐页遍历 PPT，提取每一页中的文本、表格和图片内容
    for slide_number, slide in enumerate(prs.slides, start=1):
        print(f'slide-->{slide}')
        sorted_shapes = sorted(slide.shapes, key=lambda x: (x.top, x.left))
        for a in slide.shapes:
            print(a.top)
            print(a.left)
        print(f'sorted_shapes--》{sorted_shapes}')
        for shape in sorted_shapes:
            print(f'shape--》{shape}')
            print(f'shape--》{shape.has_text_frame}')
            if shape.has_text_frame:
                # 将文本框中的文本添加到resp中，并去掉前后空格
                resp += shape.text.strip() + "\n"
                print(f'resp000--》{resp}')
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        for paragraph in cell.text_frame.paragraphs:
                            resp += paragraph.text.strip() + "\n"
                print(f'resp111--》{resp}')
            if shape.shape_type == 13:  # 13 表示图片
                image = Image.open(BytesIO(shape.image.blob))
                result, _ = ocr(np.array(image))
                if result:
                    ocr_result = [line[1] for line in result]
                    resp += "\n".join(ocr_result)
                print(f'resp222--》{resp}')
            elif shape.shape_type == 6:  # 6 表示组合
                print(f'da')
        print('*'*80)
if __name__ == '__main__':
    # 测试：替换为本地 PPT 文件路径后运行
    ppt2text(filepath='/Users/ligang/PycharmProjects/LLM/EducationRAG/data/01.pptx')