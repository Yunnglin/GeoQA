import json
import logging
import os
import random
import re
from collections import defaultdict, OrderedDict

import jieba

from utils.json_io import write_json, read_json
from utils.logger import setup_logging

EN_SYMBOLS = ',.!?[]()<>\'\'\"\"'
ZH_SYMBOLS = '，。！？【】（）《》‘’“”'

STOPWORDS = ['图示', '我国', '图中', '国家']  # 停用词


def preprocess(sentence: str) -> str:
    """
    对句子进行预处理，去掉空格和英文符号转中文
    :param sentence:
    :return: 处理以后的新句子（str是不可变的，只能返回新对象）
    """
    # 英文符号转中文
    trans = str.maketrans(EN_SYMBOLS, ZH_SYMBOLS)
    sentence = sentence.translate(trans)
    # 使用正则去掉停止符号
    return re.sub(r'[\s]', "", sentence)


def get_stopwords(file_path) -> list:
    """
    从停用词文件中读取停用词
    :param file_path: 停用词文件路径
    :return: 停用词列表
    """
    stopwords = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords


class ProcessedData:
    def __init__(self, question, question_label, background, background_label):
        self.question = question
        self.question_label = question_label
        self.background = background
        self.background_label = background_label


class Word:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end

    def __eq__(self, other):
        return type(self) == type(other) and self.text == other.text


class RawProcessor:
    def __init__(self, file_paths: [str], store_path: str, least_word_len: int = 2, use_cut: bool = False):
        # 处理过程中的词频统计 {word : freq}
        self.words_counter = defaultdict(int)
        # 统计总长度，设置合适max_len {len : freq}
        self.length_counter = defaultdict(int)
        # 要处理的文件集合的路径
        self.file_paths = file_paths
        # 处理完成的数据存放路径
        self.store_path = store_path
        # 处理完的word集合
        self.processed_data: [ProcessedData] = []
        # 是否使用分词
        self.use_cut = use_cut
        # 抽取的最小词长
        self.least_word_len = least_word_len

    def _get_same_word(self, source: str, target: str):
        """
        遍历原句子，在原句子中提取与目标句子相同部分，按照最长匹配原则
        :param source: 原句子
        :param target: 目标句子
        :return: word, word_index
        """
        source_len = len(source)
        res_words: [Word] = []  # 返回结果
        index = 0
        while index < source_len:
            # 不是标点符号，且在解析中
            if source[index] not in ZH_SYMBOLS and source[index] in target:
                word_len = 1
                # 向后延申
                while index + word_len < source_len and source[index:index + word_len + 1] in target:
                    if source[index + word_len] not in ZH_SYMBOLS:
                        word_len += 1
                    else:
                        break

                word = source[index:index + word_len]
                if len(word) >= self.least_word_len and word not in STOPWORDS:
                    # 加入该词
                    res_words.append(Word(text=word, start=index, end=index + word_len))
                    # 统计词频
                    self.words_counter[word] += 1
                index += word_len
            else:
                index += 1
        return res_words

    def _get_same_words_with_cut(self, source: str, target: str):
        """
        使用结巴分词来抽取相同词
        """
        res_words: [Word] = []
        s1_list = list(jieba.tokenize(source))
        s2_list = list(jieba.tokenize(target))
        print(' '.join(s1_list))
        print(' '.join(s2_list))
        return [word for word in s1_list if word in s2_list and word not in ZH_SYMBOLS]

    @staticmethod
    def generate_tags(sequence_len: int, words: [Word]):
        """
        根据长度和位置索引产生BIO标签
        :param sequence_len: 长度
        :param words: 抽取词集合
        :return: 标签
        """
        tags = ['O'] * sequence_len
        for word in words:
            start = word.start
            tags[start] = 'B'
            while start < word.end - 1:
                start += 1
                tags[start] = 'I'
        return tags

    def process_raw_data(self):
        """
        处理原始数据
        """
        # 循环读取文件
        for file_path in self.file_paths:
            # 读取raw_data
            raw_data = read_json(file_path)
            logger.info(f"Processing {file_path} - Question count: {len(raw_data)}")

            background_key = 'scenario_text' if "websoft" in file_path else 'background'
            # 处理raw_data
            for idx, total_question in enumerate(raw_data):
                # 提取背景和解析
                background = total_question[background_key]
                explain = total_question['explanation']
                question = total_question['question']
                # 预处理
                background = preprocess(background)
                explain = preprocess(explain)
                question = preprocess(question)
                # 跳过没有背景信息或解析的题目
                if not background or not explain:
                    continue

                # 寻找重合词
                if self.use_cut:
                    words_background = self._get_same_words_with_cut(background, explain)
                    words_question = self._get_same_words_with_cut(question, explain)
                else:
                    words_background = self._get_same_word(background, explain)
                    words_question = self._get_same_word(question, explain)
                # 生成标签
                tags_back = self.generate_tags(len(background), words_background)
                tags_question = self.generate_tags(len(question), words_question)
                # 统计长度
                self.length_counter[len(question) + len(background)] += 1
                # 添加处理好的数据
                self.processed_data.append(ProcessedData(question=question,
                                                         question_label=tags_question,
                                                         background=background,
                                                         background_label=tags_back))
                if idx < 5:
                    logger.info(f"\t example_{idx + 1} - total len: {len(question) + len(background)}")
                    logger.info(f"question_len: {len(question)}, question: {question}")
                    logger.info("words_question: " + ' | '.join([word.text for word in words_question]))
                    logger.info("tags_question: " + ' '.join(tags_question))
                    logger.info(f"background_len: {len(background)}, background: {background}")
                    logger.info("tags_back: " + ' '.join(tags_back))
                    logger.info("words_back: " + ' | '.join([word.text for word in words_background]))
                    logger.info("explain: " + explain)

    def write_processed_data(self, data_type: str, processed_data: [ProcessedData]):
        """
        将处理好的数据写入文件
        :param data_type: 写入文件的种类：train dev test
        :param processed_data: 处理好的数据
        """
        with open(os.path.join(self.store_path, f'{data_type}.txt'), 'w', encoding='utf8') as f:
            for data in processed_data:
                f.write('-DOC_START-\n')
                for i in range(len(data.question)):
                    f.write(data.question[i] + ' ' + data.question_label[i] + '\n')
                f.write('\n')
                for i in range(len(data.background)):
                    f.write(data.background[i] + ' ' + data.background_label[i] + '\n')

    def prepare_data(self, data_split_dict):
        """
        按照一定的比例将处理好的数据写入指定文件
        :param data_split_dict:
        """
        # random.shuffle(processed_data)
        total_size = len(self.processed_data)
        train_size = int(data_split_dict['train'] * total_size)
        dev_size = int(data_split_dict['dev'] * total_size)
        # [a,b)左开右闭区间
        self.write_processed_data('train', self.processed_data[:train_size])
        self.write_processed_data('dev', self.processed_data[train_size:dev_size + train_size])
        self.write_processed_data('test', self.processed_data[dev_size + train_size:])

        logger.info(f"Prepared: total size = {total_size} | train size = {train_size} | dev size = {dev_size}")


def write_counter(counter: dict, path, key, reverse=False):
    """
    将词频统计排序然后写入json文件
    """
    ordered_words_counter = OrderedDict(
        sorted(counter.items(), key=key, reverse=reverse))
    write_json(path, ordered_words_counter)


if __name__ == '__main__':
    if os.path.exists('./logs/data_info.log'):
        os.remove('./logs/data_info.log')

    setup_logging(default_path='./utils/logger_config.yaml')
    logger = logging.getLogger("data_logger")

    STOPWORDS.extend(get_stopwords('./data/stopwords.txt'))
    # 数据存储路径
    data_process_type = 'data_no_graph'
    data_path = f'./data/raw/{data_process_type}'
    store_data_path = f'./data/processed/{data_process_type}'
    # 待处理文件集合
    files = ['53_data.json', 'spider_data.json', 'websoft_data.json']
    file_paths = []
    for file in files:
        file_paths.append(os.path.join(data_path, file))

    processor = RawProcessor(file_paths=file_paths, store_path=store_data_path)
    # 处理数据
    processor.process_raw_data()
    # 写入数据
    processor.prepare_data(data_split_dict={'train': 0.7, 'dev': 0.2, 'test': 0.1})
    # 写入统计词频
    write_counter(processor.words_counter, os.path.join(store_data_path, 'word_count.json'),
                  key=lambda kv: (kv[1], kv[0]), reverse=True)
    write_counter(processor.length_counter, os.path.join(store_data_path, 'length_count.json'), key=None)
