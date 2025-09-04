import tiktoken
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import os
import time
import regex as re
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Process, Queue


def trainBPE(input_path: str | os.PathLike,
             vocab_size: int,
             special_tokens: list[str],):
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    # 1. 将special_tokens添加到词汇表中
    vocab = {idx: special_token.encode("utf-8")
             for idx, special_token in enumerate(special_tokens)}
    vocab.update({len(special_tokens)+i: bytes([i]) for i in range(256)})
    print(vocab)
    # 2. 首先读取文件，调用分块代码找出每个块边界
    # 3. 多进程处理块边界，每个进程对块进行pre_tokenization，把词数量的数据通过消息队列发送给主进程
    # 4. 迭代合并增加词汇
    pass


def BPE_train_example():

    def merge(word_freq: dict[tuple, int], pair: tuple[bytes, bytes], new_token: bytes):
        # 遍历每个word，进行merge
        new_word_freq = defaultdict(int)
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freq[tuple(new_word)] = freq
        return new_word_freq

        # 在一个简单的例子上完成BPE训练
    text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

    # 1. 初始化词表
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    vocab[0] = "<|endoftext|>".encode("utf-8")
    vocab.update(dict({1+i: bytes([i]) for i in range(256)}))
    # 2. 预分词，使用空白字符分词获取词频率表，词以bytes元组表示
    words = text.split()
    word_freq = defaultdict(int)
    for word in words:
        word_freq[tuple((bytes([byte]) for byte in word.encode("utf-8")))] += 1
    # 3. 进行merge
    num_merges = 6
    near_bytes_freq = defaultdict(int)
    vocab_len = len(vocab)
    for i in range(num_merges):
        # 3.1 遍历全部词，找到相邻两个字节出现次数最多的
        for word, freq in word_freq.items():
            for byte1, byte2 in zip(word, word[1:]):
                # 这里不能将两个bytes拼接之后放进字典，因为不知道是两个bytes是怎样的
                near_bytes_freq[(byte1, byte2)] += freq
        # 3.2 出现次数最多的选择出来作为new_token，加入到词汇表和merges中，随后进行merge
        # max函数key返回一个元组max函数会依次比较元组里面的每个元素对应的值
        max_bytes = max(near_bytes_freq.keys(),
                        key=lambda key: (near_bytes_freq.get(key), key[0]+key[1]))
        new_token = max_bytes[0]+max_bytes[1]
        vocab[vocab_len+i] = new_token
        pair = (max_bytes[0], max_bytes[1])
        merges.append(pair)
        word_freq = merge(word_freq, pair, new_token)
        near_bytes_freq.clear()

    print(f"vocab:{vocab}")
    print(f"merges:{merges}")


def pretokenize(input_path: str, idx: int, blockes: list[int], queue: Queue, special_tokens):
    """_summary_

    Args:
        idx (int): 子进程编号
        blockes (list[int]): 子程需要处理的块
        queue (Queue): 主进程与子进程间通信的消息队列，子进程只需要向队列中插入数据
    """
    try:
        if blockes is None:
            queue.put((idx, f"worker-{idx}需要处理的块为空，退出处理", None))
            return
        print(f"worker-{idx}接收到任务，处理：{blockes}")
        with open(input_path, "rb") as f:
            # 1. 每个进程只需要处理一个块，需要首先使用special_tokens对文本进行分割成多段
            f.seek(blockes[0])
            text_to_split = f.read(blockes[1]-blockes[0]).decode("utf-8")
            blocks = [string for string in re.split(
                re.escape("|".join(special_tokens)), text_to_split)]
            if len(blocks) == 0:
                queue.put((idx, f"worker-{idx}需要处理的块为空，退出处理", None))
                return
            # 2. 分割成多段，对每段进行预分词，每段单独保存
            multi_block_word_freq = []
            for block in blocks:
                word_freq = defaultdict(int)
                for word in re.finditer(
                        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", block):

                    word_freq[tuple((bytes([byte])
                                    for byte in word.group().encode("utf-8")))] += 1
                multi_block_word_freq.append(word_freq)
            queue.put((idx, "处理成功", multi_block_word_freq))
    except Exception as e:
        queue.put((idx, f"处理过程异常:{e}", None))


def merge_blocks(idx: int, start: int, end: int, word_freqs: list[defaultdict[int]], pair: tuple[bytes, bytes], new_token: bytes, queue: Queue) -> tuple[int, int, list]:
    """_summary_
    对多个块进行merge操作
    Args:
        word_freq (list[defaultdict[int]]): 原词频表
        pair (tuple[bytes, bytes]): 需要替换的字节对
        new_token (bytes): 替换之后的新字节

    Returns:
        _type_: _description_
    """
    # 单个进程merge优化：merge时可以获取到和需要merge的bytes相关的pair，主进程整合子进程相关pair的数量
    try:
        new_word_freqs = []
        for word_freq in word_freqs:
            new_word_freq = defaultdict(int)
            for word, freq in word_freq.items():
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freq[tuple(new_word)] = freq
            new_word_freqs.append(new_word_freq)
        queue.put((idx, start, end, new_word_freqs))
    except Exception as e:
        queue.put((idx, start, end, None))


def BPE_train_merge(all_word_freq: list, pair: tuple[bytes, bytes], new_token: bytes, num_process: int):
    """
    多进程对所有块进行merge操作
    """
    # 1. 计算出每个进程需要处理的block，块数少于进程数使用块数个进程
    num_block = len(all_word_freq)
    num_process = min(num_block, num_process)
    if num_process == 0:
        print("进程数量或者总块数为0")
    block_per_process = num_block // num_process
    indices = [block_per_process*i for i in range(num_process+1)]
    indices[-1] = len(all_word_freq)
    # 2. 调用多个进程进行merge操作
    # 2.1 创建消息队列
    data_queue = Queue()
    # 2.2 创建多个进程，每个进程分配任务
    processes = [Process(target=merge_blocks,  kwargs={
        'idx': i,
        'start': indices[i],
        'end': indices[i+1],
        'word_freqs': all_word_freq[indices[i]:indices[i+1]],
        'pair': pair,
        'new_token': new_token,
        'queue': data_queue
    }) for i in range(num_process)]
    for process in processes:
        process.start()
    # 2.3 主进程将子进程返回结果更新到结果中
    finish_num = 0
    all_word_freq = []
    while finish_num < num_process:
        if data_queue.qsize() != 0:
            idx, start, end, multi_word_freq = data_queue.get()
            finish_num += 1
            if multi_word_freq is None:
                print(f"worker:{idx}merge失败")
            else:
                all_word_freq.extend(multi_word_freq)
    # 3. 返回结果
    return all_word_freq


def train_BPE(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """_summary_

    Args:
        input_path (str | os.PathLike): 数据文件路径
        vocab_size (int): 词表最大大小
        special_tokens (list[str]): 特殊tokens

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: tokenizer参数，分别为词表和merges
    """

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    # 初始化词表
    vocab.update({idx: special_token.encode("utf-8")
                 for idx, special_token in enumerate(special_tokens)})
    vocab.update({len(special_tokens)+i: bytes([i]) for i in range(256)})
    # 1. 对文档进行分块，获取到文档的块边界
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # 2. 使用多个进程并行处理，进行预分词
    # 2.1 为每个进程分配任务
    data_queue = Queue()
    processes = [Process(target=pretokenize, args=(input_path,
                                                   i, boundaries[i:min(i+2, len(boundaries))] if i < len(boundaries)-1 else None, data_queue), kwargs={
                                                       'special_tokens': special_tokens
    }) for i in range(num_processes)]
    # 2.2 启动子进程
    for process in processes:
        process.start()
    # 2.3 主进程监听消息队列，从消息队列中持续取出元素,子进程处理完成后会放入None，
    finish_num = 0
    all_word_freq = []
    while finish_num < num_processes:
        if data_queue.qsize() != 0:
            idx, msg, multi_word_freq = data_queue.get()
            finish_num += 1
            if multi_word_freq is None:
                print(f"worker:{idx}处理失败,info:{msg}")
            else:
                all_word_freq.extend(multi_word_freq)
    # 3. 获取到全部分块的预分词结果之后（每个单词以及出现频率）迭代进行merge
    vocab_len = len(vocab)
    i = 0
    near_bytes_freq = defaultdict(int)
    while len(vocab) < vocab_size:
        # 3.1 将所有分块作为一个整体找出相邻两个最多的字节，然后每块单独进行合并
        # 每次重新遍历分块都需要将统计数据清零
        near_bytes_freq.clear()
        for idx, block_word_freq in enumerate(all_word_freq):
            # 统计分块的相邻字节出现频率
            for word, freq in block_word_freq.items():
                for byte1, byte2 in zip(word, word[1:]):
                    near_bytes_freq[(byte1, byte2)] += freq
        # 取出出现频率最高的字节对
        max_near_bytes = max(near_bytes_freq.keys(),
                             key=lambda key: (near_bytes_freq.get(key), key[0]+key[1]))
        new_token = max_near_bytes[0]+max_near_bytes[1]
        vocab[vocab_len+i] = new_token
        i += 1
        merges.append(max_near_bytes)
        print(max_near_bytes)
        all_word_freq = BPE_train_merge(
            all_word_freq, max_near_bytes, new_token, num_processes)
    return (vocab, merges)


if __name__ == "__main__":
    # BPE_train_example()
    input_path = "C:/Projs/assignment1-basics/tests/fixtures/tinystories_sample.txt"
    # input_path = "C:/Projs/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    vocab, merges = train_BPE(input_path, 600, ["<|endoftext|>"])
    print(sorted([vocab.get(i)
          for i in range(257, len(vocab))]))
