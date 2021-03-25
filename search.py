import collections
import json
import math
import os

import jieba

from config import get_argparse
from predict import Predict
from raw_data_process import get_stopwords
from test import ModelInfo
from utils.json_io import read_json, write_json


class Searcher:
    def __init__(self, use_geo_vocabu: bool, use_cut=False):
        if not use_geo_vocabu:
            cut_words_path = './data/all_kng_kb/kb_cut_kng.json'
        else:
            cut_words_path = './data/all_kng_kb/kb_cut_use_geoV_kng.json'
        not_cut_path = './data/all_kng_kb/kb_not_cut_kng.json'
        if use_cut:
            self.word_count_path = './data/processed/data_all/cut/redundant/word_count.json'
            self.mapping_path = './data/cut_mapping.json'
        else:
            self.word_count_path = './data/processed/data_all/no_cut/redundant/word_count.json'
            self.mapping_path = './data/no_cut_mapping.json'

        self.use_cut = use_cut
        self.use_search_cut = False
        self.use_geo_vocabu = use_geo_vocabu
        self.stop_words = set()
        print('加载停用词表...')
        self.stop_words = get_stopwords('./data/stopwords.txt')
        print('加载知识库...')
        self.kb_cut_dic = read_json(cut_words_path)
        self.kb_raw_dic = read_json(not_cut_path)
        print('加载词频统计')
        self.word_count_dic = self._load_word_count()
        print('加载关键词映射...')
        self.mapping = self._load_mapping()
        print('加载关键词抽取模型')
        self.predict = self._load_predict()

        if use_geo_vocabu:
            jieba.load_userdict('./data/geo_words_no_normal.txt')

    def _load_word_count(self) -> dict:
        """
        频数信息转为 词频-逆向文件频率
        """
        word_count_dic = read_json(self.word_count_path)
        word_no_redundant = read_json(self.word_count_path.replace('redundant', 'no_redundant'))
        total_terms, total_docs = 0, 84370
        for k in word_count_dic:
            total_terms += word_count_dic[k]
        # 计算TF-IDF
        TF_IDF = lambda terms, docs: (terms / total_terms) * (math.log10(total_docs / (docs + 1)))
        # 不存在时默认值 频率为1
        new_dict = collections.defaultdict(lambda: TF_IDF(1, 0))
        for k in word_count_dic:
            new_dict[k] = TF_IDF(word_count_dic[k], word_no_redundant[k])
        return new_dict

    def _generate_mapping(self):
        """
        不存在映射时生成，并写入json
        """
        mapping = {}
        kb_dict = self.kb_cut_dic if self.use_cut else self.kb_raw_dic
        for count, key in enumerate(self.word_count_dic):
            mapping_ids = []
            for _id, words in kb_dict.items():
                if key in words:
                    mapping_ids.append(_id)
            mapping[key] = mapping_ids
        write_json(file_path=self.mapping_path, data=mapping)

    def _load_mapping(self) -> dict:
        if not os.path.exists(self.mapping_path):
            self._generate_mapping()
        return read_json(self.mapping_path)

    def _load_predict(self) -> Predict:
        self.args = get_argparse().parse_args()
        self.args.device = '-1'
        if self.use_cut:
            model = ModelInfo('bert_base_chinese',
                              'bert_base_chinese-lstm-crf-cut-redundant-epoch_9.bin',
                              '/home/data_ti5_d/maoyl/GeoQA/save_model')
        else:
            model = ModelInfo('bert_base_chinese',
                              'bert_base_chinese-lstm-crf-no_cut-redundant-epoch_9.bin',
                              '/home/data_ti5_d/maoyl/GeoQA/save_model')
        self.args.bert_path = os.path.join('./bert', model.bert_type)
        model_path = os.path.join(model.path, model.name)
        print('model_path: ' + model_path)
        return Predict(self.args, model_path)

    def search_one(self, qa_str):
        ques_an_after = []
        if self.use_search_cut:
            k_words_list = jieba.lcut_for_search(qa_str)
        else:
            k_words_list = jieba.lcut(qa_str)
        for cut_w in k_words_list:
            if cut_w not in self.stop_words:
                ques_an_after.append(cut_w)

        if len(ques_an_after) > 0:
            pass

    def keywords_search(self, keywords):
        pass

    def get_keywords(self, question, background):
        keywords = self.predict(question, background)
        return set([key['text'] for key in keywords['in_question'] + keywords['in_background']])


class Converter:
    def __init__(self, data_path, has_answer):
        self.data_path = data_path
        self.has_answer = has_answer
        self.processed_data = []
        self.options = ['A', 'B', 'C', 'D']

    def process_data(self):
        raw_data = read_json(self.data_path)
        for ques_dic in raw_data:
            self.processed_data.append(self.get_ques_option(ques_dic))

    def search_in_kb(self):
        searcher = Searcher(use_geo_vocabu=True, use_cut=True)
        for ques_dic in self.processed_data:
            keywords = searcher.get_keywords(ques_dic['question'], ques_dic['background'])
            for option in self.options:
                knowledge = searcher.search_one(ques_dic[option])

    def get_ques_option(self, ques_dic):
        """
        拼接问题与选项
        """
        ques_option = {}

        q_str = ques_dic['question'].replace('\n', '')
        if self.has_answer:
            true_an = ques_dic['answer']
        else:
            true_an = 'G'
        ques_option['true'] = true_an

        ques_option['question'] = q_str
        ques_option['background'] = ques_dic['background'].replace('\n', '')

        for option in self.options:
            for op in [option, option.lower(), 'option' + option, 'option' + option.lower()]:
                if op in ques_dic:
                    option_str = ques_dic[op].replace('\n', '')
                    ques_option[option] = q_str + ':' + option_str
                    break
        return ques_option


if __name__ == '__main__':
    # searcher = Searcher(use_geo_vocabu=True, use_cut=False)
    converter = Converter(data_path='./data/raw/data_all/53_data.json', has_answer=True)
    converter.process_data()
    converter.search_in_kb()
