import json
import re
import jieba

E_symbols = u',.!?[]()<>'
C_symbols = u'，。！？【】（）《》'


def preprocess(sentence: str):
    # 中英符号转换
    trans = str.maketrans(E_symbols, C_symbols)
    sentence = sentence.translate(trans)
    return re.sub(r'[\s]', "", sentence)


def read_json_data(file_path) -> dict:
    with open(file_path, 'r', encoding='utf8') as f:
        f_dict = json.load(f)
        return f_dict


def get_same_word(s1: str, s2: str):
    """
    提取两个字符串的相同部分，尽量长
    :param s1: 句子1
    :param s2: 句子2
    :return:
    """
    # 处理换行空格等
    s1 = preprocess(s1)
    s2 = preprocess(s2)

    index = 0
    str_len = len(s1)
    res = list()
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
            if lens >= 2:
                res.append(s1[index:index + lens])
            index += lens
        else:
            index += 1
    return res


def get_words_with_jieba(s1: str, s2: str):
    # 处理换行空格等
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    s1_list = list(jieba.cut(s1))
    s2_list = list(jieba.cut(s2))
    print(' '.join(s1_list))
    print(' '.join(s2_list))
    return [word for word in s1_list if word in s2_list and word not in C_symbols]


def tagging(cut=False):
    dicts = read_json_data('../data/data_all/53_data.json')
    for question in dicts:
        background = question['background']
        explain = question['explanation']
        if cut:
            words = get_words_with_jieba(background, explain)
        else:
            words = get_same_word(background, explain)
        print(words)


tagging(False)
