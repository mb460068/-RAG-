from typing import TYPE_CHECKING

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    """
    创建 OCR 识别引擎实例。

    参数:
        use_cuda: 是否优先使用 GPU 推理；没有 GPU 或依赖未安装时会自动回退到 CPU 版本。
    返回:
        RapidOCR 实例，可直接调用进行图片文字识别。
    """
    try:
        # 优先使用 Paddle 版：支持 GPU 加速，识别速度更快
        from rapidocr_paddle import RapidOCR
        # det/cls/rec 分别对应检测、方向分类、识别三个模型，统一设置是否使用 GPU
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        # 未安装 GPU 版依赖时，回退到 CPU 版 rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
