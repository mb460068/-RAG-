# 文档的加载和分割
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 导包
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base.config import config as conf
from base.logger import logger

# 定义支持的文件类型应该需要的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # pptx 使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # png使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


# 定义加载函数
def load_documents_from_directory(directory_path):
    # 定义加载后的文档存储列表
    documents = []
    # 从 document_loaders中提取所有键 用来获取支持的文件名扩展集合
    supported_extensions = document_loaders.keys()
    # 定义文件相对路径
    base_path = os.path.abspath(directory_path)
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"文档目录不存在：{base_path}（当前工作目录：{os.getcwd()}）")
    # 递归遍历目标及所有子目录 获取当前目录及资源文档的路径
    for root, _, files in os.walk(base_path):
        # 利用相对路径提取分类
        rel_path = os.path.relpath(root, base_path)
        if rel_path == ".":
            source = "unknown"  # 最外层 data 文件夹兜底
        else:
            # 取相对路径的第一级文件夹
            top_path = rel_path.split(os.sep)[0]
            source = top_path.replace("_data", "") if top_path.endswith("_data") else top_path

        # 获取绝对路径进行数据加载
        for file in files:
            file_path = os.path.join(root, file)
            # 获取文件的扩展名
            file_extension = os.path.splitext(file_path)[1].lower()
            # 获取论文的题目名
            title = os.path.splitext(file)[0]
            # 如果文件类型 在 document_loaders中，就加载文件
            if file_extension in supported_extensions:
                try:
                    # 记录开始加载文件日志
                    logger.info(f"开始加载文件,{file_path}")
                    # 根据扩展名获取加载器
                    loader_class = document_loaders[file_extension]
                    # 实例化加载器
                    # 判断是否是 .txt加载器，如果是需要指定中文防止乱码
                    if file_extension == '.txt':
                        loader = loader_class(file_path, encoding='utf-8')
                    else:
                        loader = loader_class(file_path)
                    # 调用加载器的方法 加载文件内容
                    loader_docs = loader.load()
                    # 为每个加载的文件添加元数据
                    for doc in loader_docs:
                        # 添加文件末尾的分类类别 例如 lx
                        doc.metadata["source"] = source
                        # 添加文件路径
                        doc.metadata["file_path"] = file_path
                        # 添加论文题目
                        doc.metadata["title"] = title
                        # 加载时间戳
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    # 将添加元数据后的文档列表合并到总文档列表
                    documents.extend(loader_docs)
                    # 记录加载成功的日志
                    logger.info(f"加载文件成功！文件为{file_path}")

                # 如果加载失败
                except Exception as e:
                    logger.error(f"加载文件{file_path}失败，失败原因是{e}")
            # 如果文件类型不在document_loaders中，提示用户不支持文件类型
            else:
                logger.warning(f"不支持的文件类型{file_extension}")
    # 返回所有加载成功的文档列表
    return documents


# 定义切割函数
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE, child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    # 调用加载函数，从目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档数量
    logger.info(f"文档加载成功！加载到了{len(documents)}个文档")
    if not documents:
        logger.warning(f"目录中没有加载到支持的文档：{os.path.abspath(directory_path)}")
        return []
    # 初始化通用文档切分器
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 md文档切分器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化子块列表
    child_chunks = []

    # 遍历每个文档，生成唯一ID
    for i, doc in enumerate(documents):
        # 获取文件扩展名 从文档对象的元数据中取出文件路径
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        # 标价是否是 md文件
        is_markdown = (file_extension == ".md")
        # 根据文档类型选择切分器
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        # 记录日志
        logger.info(f'处理文档：{doc.metadata.get("file_path", "")}')

        # 使用切分器 切分父文档 split_documents()要求传入的数据是列表
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # 使用切分器，切分子文档
        for j, parent_doc in enumerate(parent_docs):
            # 生成父块ID
            parent_id = f"doc{i}_parent_{j}"
            # 将 父块ID加入到元数据中
            parent_doc.metadata["parent_id"] = parent_id
            parent_doc.metadata["parent_content"] = parent_doc.page_content
            # 将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # 遍历子块
            for k, sub_chunk in enumerate(sub_chunks):
                # 为子块添加元数据
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 生成子块id
                sub_chunk.metadata["id"] = f'{parent_id}_child_{k}'
                # 添加子块到总列表中
                child_chunks.append(sub_chunk)
        # 记录总块数
        logger.info(f"子块数量{len(child_chunks)}")
    # 返回子块列表
    return child_chunks


if __name__ == '__main__':
    # 调用文档处理函数
    chunks = process_documents(
        directory_path=os.path.join(conf.PROJECT_ROOT, 'data'),  # 基于项目根目录定位，避免受 IDE 工作目录影响
        parent_chunk_size=conf.PARENT_CHUNK_SIZE,  # 父块长度
        child_chunk_size=conf.CHILD_CHUNK_SIZE,  # 子块长度
        chunk_overlap=conf.CHUNK_OVERLAP  # 块重叠度
    )

    # 打印切分结果 验证子块数量 和 结构。
    print(f'切分完成的子块总数: {len(chunks)}')
    print(f'前1个子块详情: ')
    print(chunks[0] if chunks else "未生成任何子块")
