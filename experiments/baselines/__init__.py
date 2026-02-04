"""
基线方法实现
============

实现论文中的对比基线方法:
- RandomFuzzer: 随机生成payload
- GrammarFuzzer: 基于语法的生成
- GrammarRL: 基于语法的强化学习 (不使用语言模型)
"""

from .random_fuzzer import RandomFuzzer
from .grammar_fuzzer import GrammarFuzzer, GrammarRL

__all__ = [
    "RandomFuzzer",
    "GrammarFuzzer",
    "GrammarRL",
]
