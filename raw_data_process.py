import json
import re
from collections import defaultdict

import jieba

E_symbols = u',.!?[]()<>'
C_symbols = u'，。！？【】（）《》'

STOPWORDS = ['图示', '我国', '图中', '国家']

SINGAL_DICT = defaultdict(int)  # 统计单字


def preprocess(sentence: str) -> str:
    """
    对句子进行预处理，去掉空格和中英符号转换
    :param sentence:
    :return: 处理以后的句子
    """
    # 中英符号转换
    trans = str.maketrans(E_symbols, C_symbols)
    sentence = sentence.translate(trans)
    return re.sub(r'[\s]', "", sentence)


def read_json_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        f_dict = json.load(f)
        return f_dict


def get_stopwords(file_path) -> list:
    stopwords = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords


def get_same_word(s1: str, s2: str, word_len=1):
    """
    提取两个字符串的相同部分，尽量长
    :param word_len: 标注的连续词语长度
    :param s1: 句子1
    :param s2: 句子2
    :return:
    """
    index = 0
    str_len = len(s1)
    res = list()
    res_index = list()
    while index < str_len:
        # 不是标点符号，且在解析中
        if s1[index] not in C_symbols and s1[index] in s2:
            lens = 1
            # 向后延申
            while index + lens < str_len and s1[index:index + lens + 1] in s2:
                if s1[index + lens] in C_symbols:
                    break
                lens += 1
            # 　未被停用
            word = s1[index:index + lens]
            if len(word) >= word_len and word not in STOPWORDS:
                res.append(word)
                res_index.append([index, index + lens])
                # 统计单字词频
                if len(word) == 1:
                    SINGAL_DICT[word] += 1
            index += lens
        else:
            index += 1
    return res, res_index


def get_words_with_jieba(s1: str, s2: str):
    s1_list = list(jieba.cut(s1))
    s2_list = list(jieba.cut(s2))
    print(' '.join(s1_list))
    print(' '.join(s2_list))
    return [word for word in s1_list if word in s2_list and word not in C_symbols]


def generate_tags(n, words_index):
    """
    根据长度和位置索引产生标签
    :param n: 长度
    :param words_index: 位置索引
    :return: 标签
    """
    tags = ['O'] * n
    for index in words_index:
        start = index[0]
        tags[start] = 'B'
        while start < index[1] - 1:
            start += 1
            tags[start] = 'I'
    return tags


def tagging(files=[], cut=False):
    """
    根据最长匹配方法产生BIO标签
    :param files: 文件集合
    :param cut: 是否使用jieba分词
    :return: None
    """
    dicts = []
    for file in files:
        dicts.extend(read_json_data(file))
    with open('data/train.txt', 'w', encoding='utf8') as f:
        for idx, total_question in enumerate(dicts):
            # 提取背景和解析
            background_keys = ['background', 'scenario_text']
            for k in background_keys:
                if k in total_question:
                    background = total_question[k]
                    break
            explain = total_question['explanation']
            question = total_question['question']
            # 预处理
            background = preprocess(background)
            explain = preprocess(explain)
            question = preprocess(question)
            # 寻找重合词
            words_back, words_index_back = get_same_word(background, explain, word_len=2)
            words_question, words_index_question = get_same_word(question, explain, word_len=2)
            # 生成标签
            tags_back = generate_tags(len(background), words_index_back)
            tags_question = generate_tags(len(question), words_index_question)
            if idx < 10:
                print(f"--example_{idx + 1}--")
                print("question: " + question)
                print("words_question: " + ' | '.join(words_question))
                print("tags_question: " + ' '.join(tags_question))
                print("background: " + background)
                print("tags_back: " + ' '.join(tags_back))
                print("words_back: " + ' | '.join(words_back))
                print("explain: " + explain)

            f.write('-DOC_START-\n')
            for i in range(len(question)):
                f.write(question[i] + ' ' + tags_question[i] + '\n')
            f.write('\n')
            for i in range(len(background)):
                f.write(background[i] + ' ' + tags_back[i] + '\n')

            # f.write(' '.join(background) + '|||' + ' '.join(tags) + '\n')


if __name__ == '__main__':
    STOPWORDS.extend(get_stopwords('./data/stopwords.txt'))
    file_paths = ['./data/raw/data_all/53_data.json', './data/raw/data_all/websoft_data.json']
    tagging(file_paths, cut=False)
    # print(sorted(SINGAL_DICT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    # print(type(read_json_data('data/raw/data_all/53_data.json')))
