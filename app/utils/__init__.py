#!/usr/bin/env python
# coding:utf8

from pathlib import Path
from typing import Callable, Optional, Generator


class FolderWalker:
    """
    文件夹遍历器
    """
    __slots__ = ['root_dir', '_root_path']

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._root_path: Path = Path(self.root_dir)

    def simple_walk(
        self, name_filter: Optional[Callable]=None, only_file: bool=True
    ) -> Generator:
        """
        遍历文件夹，返回文件路径及更新时间

        ps: 更新时间可以作为文件指纹, 决定是否重新同步文件

        >>> list(FolderWalker('/tmp').simple_walk(name_filter=lambda f: any(f.endswith(x) for x in ['.pdf', '.txt', '.md'])))
        [(PosixPath('/tmp/a.txt'), 1724224576.604822)]
        """
        for path in self._root_path.rglob('*'):
            if only_file and not path.is_file():
                continue

            if name_filter is None or name_filter(path.name):
                # _s_path: str = str(path)
                _ts: float = path.stat().st_mtime
                yield path, _ts


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
