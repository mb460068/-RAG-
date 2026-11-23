# -RAG-
本项目是基于 FastAPI 的教育类资料智能问答系统。后端结合 MySQL（存会话/FAQ）、Redis（缓存）与 Milvus（向量库），查询采用“BM25+混合检索”策略：优先命中 FAQ，未命中则触发 RAG 流程，结合 BGE-M3、Reranker 重排与查询分类器，调用 LLM 生成答案。系统支持多格式文档及图片 OCR 解析。前端实现 WebSocket 流式输出、Markdown 渲染与多维度筛选，提供流畅的智能问答体验。
