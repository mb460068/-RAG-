# 该文件用于帮助 rag_qa这个Python包，管理包内的脚本，方便其它包的调用。

import os, sys

# 1. 记录当前目录。
current_dir = os.path.abspath(__file__)


rag_qa_path = os.path.dirname(current_dir)


sys.path.insert(0, rag_qa_path)


from core.prompts import RAGPrompts
from core.vector_store import VectorStore

from core.rag_system import RAGSystem