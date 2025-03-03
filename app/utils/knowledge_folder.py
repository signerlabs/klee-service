#!/usr/bin/env python
# coding:utf8

from pathlib import Path
from typing import Generator
import aiofiles.os

from app.utils import FolderWalker


def walk_knowledge_folder(
    knowledge_folder: str,
    ext_filter: tuple=('.pdf', '.txt', '.md', '.docx', '.pptx', '.xlsx')
) -> Generator:
    """
    遍历知识库文件夹中的有效文件

    >>> [(str(p), ts) for p, ts in walk_knowledge_folder('/tmp')]
    [('/tmp/a.txt', 1724224576.604822)]
    """
    walker = FolderWalker(knowledge_folder)

    if not ext_filter:
        return walker.simple_walk()

    return (
        (path, ts) for path, ts in walker.simple_walk() if path.suffix in ext_filter
    )


def gen_knowledge_cache_path(
    folder_path: str, knowledge_id: str, cache_sub:str='.klee-cache'
) -> Path:
    """
    根据 knowledge_id 及其关联的 folder_path，获取知识库缓存目录结构

    示例：/path/to/<knowledge_foler>/.klee-cache/.7B043300-1CE1-4B61-8798-3DF9D05F2169
    """
    cache_base = Path(folder_path) / cache_sub

    _knowledge_id = f'.{knowledge_id}'

    return cache_base / _knowledge_id

async def init_folder_knowledge_cache(
    folder_path: str, knowledge_id: str, cache_sub: str='.klee-cache'
) -> Path:
    """
    初始化文件夹知识库缓存目录
    """
    knowledge_cache = gen_knowledge_cache_path(folder_path, knowledge_id, cache_sub)

    await aiofiles.os.makedirs(knowledge_cache, exist_ok=True)
    # touch 一个索引为 <knowledge_id>.0 的文件
    # await aiofiles.open(knowledge_cache / f'{knowledge_id}.0', mode='a').close()
    return knowledge_cache

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
