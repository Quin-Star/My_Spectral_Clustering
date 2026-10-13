# -*- coding: utf-8 -*-
"""通用工具模块：日志、目录、随机种子、计时、json 保存。

这里只放与算法无关的工程脚手架，保证其它模块干净。
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

import numpy as np

PathLike = Union[str, Path]

def setup_console(encoding: str = "utf-8") -> None:
    """把标准输出改成 UTF-8，解决 Windows 控制台中文乱码。

    重定向到文件/管道时 Python 会用系统默认编码（中文 Windows 上是 GBK），
    中文日志就会变成乱码。老版本 Python 没有 reconfigure 方法，忽略即可。
    """
    try:
        sys.stdout.reconfigure(encoding=encoding)
    except Exception:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """创建（或复用）一个带统一格式的 logger。 日志

    重复调用同名 logger 不会重复添加 handler，靠 logger.handlers 判空实现。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False        # 不向 root logger 冒泡，避免重复输出
    return logger

def ensure_dir(path: PathLike) -> Path:
    """确保目录存在（不存在就递归创建），返回 Path 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def set_random_seed(seed: int = 17) -> None:
    """固定全局随机种子，保证实验可复现。

    注意：本项目内部一律使用 np.random.default_rng(seed) 这种独立生成器，
    这里设置全局种子只是为了兼容可能用到的第三方库（如 sklearn）。 相当于兜底
    """
    np.random.seed(seed)

class NumpyJSONEncoder(json.JSONEncoder):
    """让 json 能序列化 numpy 的标量、数组和 Path。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Any, path: PathLike, indent: int = 2) -> Path:
    """把字典写成 json 文件（自动创建父目录），返回文件路径。"""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, cls=NumpyJSONEncoder, indent=indent, ensure_ascii=False)
    return path


@contextmanager
def timer() -> Iterator[dict]:
    """计时上下文管理器。

    用法：
        with timer() as t:
            do_something()
        print(t["seconds"])
    """
    record: dict = {}
    start = time.perf_counter()             # 单调时钟，不受系统时间调整影响
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start


if __name__ == "__main__":
    setup_console()
    set_random_seed(0)
    log = get_logger("utils")
    log.info("logger 正常")
    with timer() as t:
        pass
    log.info("timer 正常：%.6f 秒", t["seconds"])
    print("save_json 正常：", save_json({"ok": True}, "outputs/_utils_check.json"))









