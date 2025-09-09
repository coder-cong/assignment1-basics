# 导入 Self 用于类型提示，List 用于清晰表示列表类型
from typing import Self, Optional, List, Iterable, Iterator, BinaryIO
import tiktoken
from abc import ABC
from dataclasses import dataclass, field
from collections import defaultdict
import os
import time
import regex as re
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Process, Queue
import logging
# 配置日志输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,  # 设置最低级别为 INFO
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # 日志输出到文件
        logging.StreamHandler()  # 日志输出到控制台
    ]
)
# 获取一个Logger实例 (推荐使用 __name__ 来区分不同模块的Logger)
logger = logging.getLogger(__name__)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    e.g. 实际的分界点在12 20 分块的点为5 10 15 20 第一块和第二块有重叠，返回了相同的分界点，经过排序去重之后结果会少
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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


@dataclass
class ProcessMessageBlock:
    # 子进程编号
    idx: int = 0
    # 子进程处理时间
    process_time: float = 0.0
    # 子进程消息
    msg: str = ""
    # 子进程处理区间起点
    start: Optional[int] = None
    # 子进程处理区间终点
    end: Optional[int] = None
    # pretokenize返回多个分块的预分词结果
    pretokenized_chunks: List[defaultdict] = None
    # BPE Train merge过程子进程返回多个分块的merge结果
    merged_chunks: List[defaultdict] = None
    # merge过程中统计相邻字节对出现数据
    near_bytes: Optional[defaultdict] = None
    # merge过程中统计新token相邻字节对数量
    new_token_freq: Optional[defaultdict] = None

    def set_msg(self, msg: str) -> Self:
        """设置消息，并返回自身以便链式调用。"""
        self.msg = msg  # 直接赋值比 __setattr__ 更常用和推荐
        return self

    def set_process_time(self, process_time: float) -> Self:
        """子进程完成任务需要时间，进行性能分析"""
        self.process_time = process_time
        return self

    def set_pretokenized_chunks(self, pretokenized_chunks: List[defaultdict]) -> Self:
        """设置预分词块，并返回自身以便链式调用。"""
        self.pretokenized_chunks = pretokenized_chunks
        return self

    def set_idx(self, idx: int) -> Self:
        """设置子进程编号，并返回自身以便链式调用。"""
        self.idx = idx
        return self

    def set_start(self, start: int) -> Self:
        """设置处理区间起点，并返回自身以便链式调用。"""
        self.start = start
        return self

    def set_end(self, end: int) -> Self:
        """设置处理区间终点，并返回自身以便链式调用。"""
        self.end = end
        return self

    def set_merged_chunks(self, merged_chunks: List[defaultdict]) -> Self:
        """设置合并后的分块，并返回自身以便链式调用。"""
        self.merged_chunks = merged_chunks
        return self

    def set_near_bytes(self, near_bytes: defaultdict) -> Self:
        """设置相邻字节对统计数据，并返回自身以便链式调用。"""
        self.near_bytes = near_bytes
        return self

    def set_new_token_freq(self, new_token_freq: defaultdict) -> Self:
        """设置新token相邻字节对统计数据，并返回自身以便链式调用。"""
        self.new_token_freq = new_token_freq
        return self


def pretokenize(input_path: str, idx: int, blockes: list[int], queue: Queue, special_tokens) -> list[defaultdict]:
    """_summary_

    Args:
        idx (int): 子进程编号
        blockes (list[int]): 子程需要处理的块
        queue (Queue): 主进程与子进程间通信的消息队列，子进程只需要向队列中插入数据
    """
    message_block = ProcessMessageBlock(idx=idx)
    try:
        if blockes is None:
            msg = f"worker-{idx} 需要处理的块为空，退出处理"
            queue.put(message_block.set_msg(msg))
            return
        logger.info(f"worker-{idx} 接收到任务，处理：{blockes}")
        with open(input_path, "rb") as f:
            # 1. 每个进程只需要处理一个块，需要首先使用special_tokens对文本进行分割成多段
            f.seek(blockes[0])
            text_to_split = f.read(blockes[1]-blockes[0]).decode("utf-8")
            blocks = [string for string in re.split(
                re.escape("|".join(special_tokens)), text_to_split)]
            if len(blocks) == 0:
                msg = f"worker-{idx} 需要处理的块为空，退出处理"
                logger.info(msg)
                queue.put(message_block.set_msg(msg))
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
            msg = "处理成功"
            queue.put(message_block.set_msg(
                msg).set_pretokenized_chunks(multi_block_word_freq))
    except Exception as e:
        msg = f"处理过程异常:{e}"
        logger.info(f"worker-{idx} {msg}")
        queue.put(message_block.set_msg(msg))


def merge(word_freq: list[defaultdict[int]], pair: tuple[bytes, bytes], new_token: bytes, related_pair_count: defaultdict[int], new_token_freq: defaultdict[int]) -> tuple[list[defaultdict[int]]]:
    """_summary_
    merge单个文档分块的词频表
    Args:
        word_freq (list[defaultdict[int]]): 单个文档分块的词频表
        pair (tuple[bytes, bytes]): 需要merge的字节对
        new_token (bytes): merge之后的新token
    """
    new_word_freq = defaultdict(int)
    for word, freq in word_freq.items():
        i = 0
        new_word = []
        while i < len(word):
            if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(new_token)
                # 统计和merge的字节对相邻的字节对
                if related_pair_count is not None:
                    if i > 0:
                        related_pair_count[(word[i-1], word[i])] += freq
                    if i+1 < len(word)-1:
                        related_pair_count[(word[i+1], word[i+2])] += freq
                i += 2
            else:
                new_word.append(word[i])
                i += 1
            # 不仅需要统计相邻的字节对，还需要统计新token相邻字节对
            last_index = len(new_word)-1
            if last_index > 0:
                if new_word[last_index-1] == new_token or new_word[last_index] == new_token:
                    new_token_freq[(new_word[last_index-1],
                                    new_word[last_index])] += freq
        new_word_freq[tuple(new_word)] = freq
    return new_word_freq


def merge_blocks(idx: int, start: int, end: int, word_freqs: list[defaultdict[int]], pair: tuple[bytes, bytes], new_token: bytes, queue: Queue) -> tuple[int, int, list]:
    """_summary_
    单个子进程进行merge操作
    Args:
        word_freq (list[defaultdict[int]]): 原词频表
        pair (tuple[bytes, bytes]): 需要替换的字节对
        new_token (bytes): 替换之后的新字节

    Returns:
        _type_: _description_
    """

    start_time = time.time()
    # 单个进程merge优化：merge时可以获取到和需要merge的bytes相关的pair，主进程整合子进程相关pair的数量，并根据这个数量修改词频统计表
    message_block = ProcessMessageBlock(idx=idx, start=start, end=end)
    # 和需要合并的pair相的字节对数量
    near_byte_count = defaultdict(int)
    # 包含新token的字节对统计
    new_token_freq = defaultdict(int)
    try:
        new_word_freqs = []
        for word_freq in word_freqs:
            new_word_freqs.append(
                merge(word_freq, pair, new_token, near_byte_count, new_token_freq))
        message_block.set_merged_chunks(
            new_word_freqs).set_near_bytes(near_byte_count).set_new_token_freq(new_token_freq)
    except Exception as e:
        msg = f"worker-{idx} 处理过程异常:{e}"
        message_block.set_msg(msg)
    end_time = time.time()
    queue.put(message_block.set_process_time(end_time-start_time))


def BPE_train_merge(all_word_freq: list, pair: tuple[bytes, bytes], new_token: bytes, num_process: int):
    """
    多进程对所有块进行merge操作
    """
    # 1. 计算出每个进程需要处理的block，块数少于进程数使用块数个进程
    num_block = len(all_word_freq)
    num_process = min(num_block, num_process)
    if num_process == 0:
        logger.info("进程数量或者总块数为0")
    block_per_process = num_block // num_process
    indices = [block_per_process*i for i in range(num_process+1)]
    indices[-1] = len(all_word_freq)
    start_time = time.time()
    # 2. 调用多个进程进行merge操作 定位到问题：每次都重新创建进程，时间开销大
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

    end_time = time.time()
    logger.info(f"多进程merge启动时间：{end_time-start_time}")
    # 2.3 主进程将子进程返回结果更新到结果中
    finish_num = 0
    all_word_freq = []
    near_bytes = defaultdict(int)
    new_token_freq = defaultdict(int)
    while finish_num < num_process:
        if data_queue.qsize() != 0:
            message_block: ProcessMessageBlock = data_queue.get()
            finish_num += 1
            if message_block.merged_chunks is None:
                msg = f"worker-{message_block.idx} merge失败"
                logger.info(msg)
            else:
                all_word_freq.extend(message_block.merged_chunks)
            sub_near_bytes = message_block.near_bytes
            if sub_near_bytes is not None:
                for word, freq in sub_near_bytes.items():
                    near_bytes[word] += freq
            sub_new_token_freq = message_block.new_token_freq
            if sub_new_token_freq is not None:
                for word, freq in sub_new_token_freq.items():
                    new_token_freq[word] += freq

    # 3. 返回结果
    return (all_word_freq, near_bytes, new_token_freq)


def BPE_train_pretokenize(input_path: str, special_tokens: list[str], num_processes: int = 4, ) -> list[defaultdict[int]]:
    """
    多进程完成BPE预分词
    """
    # 1. 对文档进行分块，获取到文档的块边界
    with open(input_path, "rb") as f:
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
            message_block: ProcessMessageBlock = data_queue.get()
            finish_num += 1
            if message_block.pretokenized_chunks is None:
                msg = f"worker-{message_block.idx} 处理失败,info:{message_block.msg}"
                logger.info(msg)
            else:
                all_word_freq.extend(message_block.pretokenized_chunks)
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
    # 1. 初始化词表
    vocab.update({idx: special_token.encode("utf-8")
                 for idx, special_token in enumerate(special_tokens)})
    vocab.update({len(special_tokens)+i: bytes([i]) for i in range(256)})
    # 2. 预分词
    start_time = time.time()
    num_processes = 4
    all_word_freq = BPE_train_pretokenize(
        input_path, special_tokens, num_processes)
    end_time = time.time()
    logger.info(f"-----预分词时间：{end_time-start_time}----")
    # 3. 获取到全部分块的预分词结果之后（每个单词以及出现频率）迭代进行merge
    vocab_len = len(vocab)
    i = 0
    near_bytes_freq = defaultdict(int)
    # 优化之后：每次完成merge不需要重新统计相邻bytes的频率，而是每次merge完成后只是对相关的进行修改即可
    # 初始统计一次
    start_time = time.time()
    for idx, block_word_freq in enumerate(all_word_freq):
        # 统计分块的相邻字节出现频率
        for word, freq in block_word_freq.items():
            for byte1, byte2 in zip(word, word[1:]):
                near_bytes_freq[(byte1, byte2)] += freq
    end_time = time.time()
    logger.info(f"-----相邻字节对频率统计时间：{end_time-start_time}-----")
    while len(vocab) < vocab_size:
        # 取出出现频率最高的字节对
        max_near_bytes = max(near_bytes_freq.keys(),
                             key=lambda key: (near_bytes_freq.get(key), key[0]+key[1]))
        new_token = max_near_bytes[0]+max_near_bytes[1]
        vocab[vocab_len+i] = new_token
        i += 1
        merges.append(max_near_bytes)
        print(max_near_bytes)
        start_time = time.time()
        all_word_freq, related_neighbor, new_token_freq = BPE_train_merge(
            all_word_freq, max_near_bytes, new_token, num_processes)
        end_time = time.time()
        logger.info(f"用时：{end_time-start_time}")
        # 主进程merge之后只对相邻的字节对统计数量进行更新
        # 1. 首先删除掉merge的字节对的频率
        near_bytes_freq.pop(max_near_bytes)
        # 2. 遍历相关的字节对，从原来的统计表中减掉
        for word, freq in related_neighbor.items():
            near_bytes_freq[word] -= freq
            if near_bytes_freq[word] == 0:
                near_bytes_freq.pop(word)
        # 3. 将新token相关的字节对更新到原来的统计表中
        near_bytes_freq.update(new_token_freq)
    return (vocab, merges)


@dataclass
class BPETokenizerParam:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass


if __name__ == "__main__":
    # BPE_train_example()
    # input_path = "C:/Projs/assignment1-basics/tests/fixtures/tinystories_sample.txt"
    input_path = "C:/Projs/assignment1-basics/tests/fixtures/corpus.en"
    # input_path = "C:/Projs/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    vocab, merges = train_BPE(input_path, 500, ["<|endoftext|>"])
    print(sorted([vocab.get(i)
          for i in range(257, len(vocab))]))
