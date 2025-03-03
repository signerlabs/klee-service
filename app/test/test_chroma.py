import chromadb
import asyncio
from chromadb.config import Settings
import time
import os
import logging
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import \
    SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
import asyncio
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
from langchain.schema.document import Document
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# llama_index import
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
# alias name
from llama_index.core.settings import Settings as llama_index_Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 配置日志记录
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

logging.disable(logging.CRITICAL)

# 获取用户主目录
user_home = os.path.expanduser("~")

# 构建数据库路径
DATABASE_FOLDER_PATH = os.path.join(user_home, 'Library/com.signerlabs.klee/chroma')
# 本地的Embeddings的model路径
SENTENCE_TRANSFORMER_MODEL_PATH = os.path.join(user_home, "Library/com.signerlabs.klee/llm/all-MiniLM-L6-v2")

collection_name = "xJGxqRT4a55y88RzmJAogQDFCpKH"


# 新增：查询文本分割函数
def split_query(query_text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=75)
    return text_splitter.split_text(query_text)

async def query_single_collection(collection_name, query_text, top_k):
    client = chromadb.PersistentClient(path=DATABASE_FOLDER_PATH, settings=Settings())

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=SENTENCE_TRANSFORMER_MODEL_PATH)
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    # logger.debug(f'查询知识库或笔记: {collection_name}')
    # 检查collection是否为空
    if collection.count() == 0:
        # logger.debug(f'知识库或笔记为空: {collection_name}')
        return {
            'documents': [[]],
            'distances': [[]],
            'ids': [[]],
            'metadatas': [[]]
        }

    # 分割查询文本
    query_chunks = split_query(query_text)

    # 对每个查询chunk进行查询
    all_results = []
    for i, query_chunk in enumerate(query_chunks):
        results = collection.query(
            query_texts=[query_chunk],
            n_results=top_k
        )
        for j, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            all_results.append({
                'id': results['ids'][0][j],
                'content': doc,
                'metadata': results['metadatas'][0][j],
                'distance': distance,
                'query_chunk_index': i
            })

    # 使用TF-IDF进行二次排序
    if all_results:
        tfidf = TfidfVectorizer().fit_transform([query_text] + [r['content'] for r in all_results])
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        # 加权分数计算
        for i, (result, similarity) in enumerate(zip(all_results, cosine_similarities)):
            chunk_index = result['metadata'].get('chunk_index', 0)
            chunk_weight = 1 / (1 + chunk_index * 0.1)  # 前面的chunk权重略高
            query_chunk_weight = 1 / (1 + result['query_chunk_index'] * 0.05)  # 查询的前面部分权重略高
            weighted_score = similarity * chunk_weight * query_chunk_weight
            all_results[i]['weighted_score'] = weighted_score

        # 排序并返回top_k结果
        sorted_results = sorted(all_results, key=lambda x: x['weighted_score'], reverse=True)
        top_results = sorted_results[:top_k]
    else:
        top_results = []

    return {
        'documents': [[r['content'] for r in top_results]] if top_results else [[]],
        'distances': [[1 - r.get('weighted_score', 0) for r in top_results]] if top_results else [[]],
        'ids': [[r['id'] for r in top_results]] if top_results else [[]],
        'metadatas': [[r['metadata'] for r in top_results]] if top_results else [[]]
    }


async def test_chroma_query():
    # 初始化Chroma客户端
    print(f'准备测试查询: {collection_name}')
    result = await query_single_collection(collection_name, "陆云是谁？", top_k=20)
    print(f'查询的结果: {result}')


def main():
    asyncio.run(test_chroma_query())


if __name__ == "__main__":
    main()
