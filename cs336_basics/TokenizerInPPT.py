import tiktoken
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import os


# 定义tokenizer接口
class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    # 基于字符分词，利用map对每个字符进行操作
    def encode(self, string: str) -> list[int]:
        indices = list(map(ord, string))
        return indices

    def decode(self, indices: list[int]) -> str:
        # 使用空字符串.join(list)完成拼接字符串
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        # string encode得到字节数组
        return list(map(int, string.encode()))

    def decode(self, indices: list[int]) -> str:
        #  bytes(list)可以将list中的每一个元素转换为字节
        string_bytes = bytes(indices)
        # 字节数组 decode可以得到字符串
        string = string_bytes.decode("utf-8")
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    # TODO 测试merge功能
    # 将原序列indices中的pair合并为new_index
    new_indices = []
    index = 0
    while index < len(indices):
        if index < len(indices)-1 and indices[index] == pair[0] and indices[index+1] == pair[1]:
            new_indices.append(new_index)
            index += 2
        else:
            new_indices.append(indices[index])
            index += 1
    return new_indices


@dataclass
class BytePairEncodingParam:
    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


class BytePairEncodingTokenizer(Tokenizer):
    def __init__(self, param: BytePairEncodingParam):
        self.param = param

    def encode(self, string: str) -> list[int]:
        # TODO 测试编码功能
        indices = list(map(int, string.encode()))
        for pair, new_index in self.param.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.param.vocab.get, indices))
        return b"".join(bytes_list).decode()


def trainBPE_origin(string: str, merge_num: int) -> BytePairEncodingParam:
    # 这种方式每次merge都需要完整遍历一遍语料库，效率较低
    indices = list(map(int, string.encode("utf-8")))
    # 定义vocab和merges，vocab定义了int到bytes的转换
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: dict[tuple[int, int], int] = {}
    for i in range(merge_num):
        # 1. 每次merge找出频率最高的一对index
        counts = defaultdict(int)
        pairs = list(zip(indices, indices[1:]))
        for pair in pairs:
            counts[pair] += 1
        max_pair = max(pairs, key=counts.get)
        # 2. 频率最高的一对index加入字典中
        new_index = 256+i
        merges[max_pair] = new_index
        vocab[new_index] = vocab[max_pair[0]]+vocab[max_pair[1]]
        # 3. 完成一次merge操作
        indices = merge(indices, max_pair, new_index)
    return BytePairEncodingParam(vocab=vocab, merges=merges)


def byteTokenizerTest():
    string = "h e l l o hellohello😊world"
    byteTokenizer = ByteTokenizer()
    encoded = byteTokenizer.encode(string)
    print(encoded)
    decoded = byteTokenizer.decode(encoded)
    print(decoded)
    print(string == decoded)


def characterTokenizerTest():
    string = "hello😊world"
    characterTokenizer = CharacterTokenizer()
    encoded = characterTokenizer.encode(string)
    print(encoded)
    decoded = characterTokenizer.decode(encoded)
    print(decoded)
    print(string == decoded)


def BPETokenizerTest():
    # 测试bpe训练
    # 1. 读取文本文件为字符串
    # 2. 调用训练函数
    # 3. 打印查看训练参数结果
    # 4. 打断点查看情况
    text = open("C:/Projs/assignment1-basics/tests/fixtures/tinystories_sample.txt",
                encoding="utf-8").read()
    # 使用一个简单字符串测试
    param = trainBPE_origin(text, 10)
    tokenizer = BytePairEncodingTokenizer(param)
    print(param.vocab)
    print(param.merges)


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


if __name__ == "__main__":
    # characterTokenizerTest()
    # byteTokenizerTest()
    BPETokenizerTest()

    # BPETrainExample()
    # trainBPE("C:/Projs/assignment1-basics/tests/fixtures/tinystories_sample.txt",
    #          vocab_size=15000, special_tokens=["<|endoftext|>"])
