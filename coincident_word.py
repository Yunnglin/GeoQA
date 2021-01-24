import json
import re
import jieba

E_symbols = u',.!?[]()<>'
C_symbols = u'，。！？【】（）《》'

STOPWORDS = ['图示', '我国', '图中']


def preprocess(sentence: str):
    # 中英符号转换
    trans = str.maketrans(E_symbols, C_symbols)
    sentence = sentence.translate(trans)
    return re.sub(r'[\s]', "", sentence)


def read_json_data(file_path) -> dict:
    with open(file_path, 'r', encoding='utf8') as f:
        f_dict = json.load(f)
        return f_dict


def get_same_word(s1: str, s2: str, word_len=3):
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
        # find_all_index = lambda sentence, character: [i for i in range(len(sentence)) if sentence[i] == character]
        # all_index = find_all_index(s2, s1[index])
        # 不是标点符号，且在解析中
        if s1[index] not in C_symbols and s1[index] in s2:
            lens = 1
            # 向后延申
            while index + lens < str_len and s1[index:index + lens + 1] in s2:
                if s1[index + lens] in C_symbols:
                    break
                lens += 1
            # 　长度大于２，且未被停用
            if lens >= word_len:
                word = s1[index:index + lens]
                if word not in STOPWORDS:
                    res.append(word)
                    res_index.append([index, index + lens])
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


def tagging(cut=False):
    """
    根据最长匹配方法产生BIO标签
    :param cut: 是否使用jieba分词
    :return: None
    """
    dicts = read_json_data('data/raw/data_all/53_data.json')
    with open('data/train.txt', 'w', encoding='utf8') as f:
        for question in dicts:
            background = preprocess(question['background'])
            explain = preprocess(question['explanation'])
            if cut:
                words, words_index = get_words_with_jieba(background, explain)
            else:
                words, words_index = get_same_word(background, explain)
            print(words)
            tags = generate_tags(len(background), words_index)
            # for i in range(len(background)):
            #     f.write(background[i] + '\t' + tags[i] + '\n')
            f.write(' '.join(background) + '|||' + ' '.join(tags) + '\n')


tagging(False)
