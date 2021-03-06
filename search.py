import collections
import json
import math
import os
import time

import jieba

from config import get_argparse
from predict import Predict
from raw_data_process import get_stopwords
from test import ModelInfo
from utils.json_io import read_json, write_json
from utils.similarity import tf_similarity, jaccard_similarity_set, text_rank_similarity, tfidf_index


def merge_dict(dict1: dict, dict2: dict, weight1=1.0, weight2=1.0, limit=None) -> dict:
    """
    合并两个dict，同时指定dict值的权重
    """
    if len(dict1) > len(dict2):
        dict1, dict2, weight1, weight2 = dict2, dict1, weight2, weight1
    # dict1 < dict2 遍历dict1
    for k in dict1:
        dict2[k] = dict2[k] * weight2 + dict1[k] * weight1
    if limit:
        return get_dict_topk(dict2, limit)
    else:
        return dict2.copy()


def get_dict_topk(origin_dict, topk) -> dict:
    new_dict = collections.defaultdict(float)
    new_dict.update(sorted(origin_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:topk])
    return new_dict


def norm_dict_value(origin_dict: dict):
    if not origin_dict.values():
        return
    max_value = max(origin_dict.values())
    for k in origin_dict:
        origin_dict[k] /= max_value


class Searcher:
    def __init__(self, use_geo_vocabu: bool, use_cut=True, use_tfidf=False):
        if not use_geo_vocabu:
            cut_words_path = './data/all_kng_kb/kb_cut_kng.json'
        else:
            cut_words_path = './data/all_kng_kb/kb_cut_use_geoV_kng.json'
        not_cut_path = './data/all_kng_kb/kb_not_cut_kng.json'

        self.word_count_path = './data/all_kng_kb/kb_word_count.json'
        self.mapping_path = './data/mappings/kb_mapping.json'
        self.use_cut = use_cut
        self.use_tfidf = use_tfidf
        self.use_search_cut = False
        self.use_geo_vocabu = use_geo_vocabu

        print('加载停用词表...')
        self.stop_words = get_stopwords('./data/stopwords.txt')
        print('加载知识库...')
        self.kb_cut_dic = read_json(cut_words_path)
        self.kb_raw_dic = read_json(not_cut_path)
        print('加载词频统计...')
        self.word_count_dic = read_json(self.word_count_path)
        self.word_weight_dic = collections.defaultdict(lambda: 1.0)
        print('加载倒排索引...')
        self.keyword_id_mapping = self._load_mapping()
        print('加载关键词抽取模型...')
        self.predict = self._load_predict()

        if use_geo_vocabu:
            jieba.load_userdict('./data/geo_words_no_normal.txt')

    def _get_sentence_tfidf(self, sentence, keywords=None) -> dict:
        """
        计算句子中每个词的tfidf，并返回字典
        """
        res = {}
        if not keywords:
            keywords = self.cut_with_stopwords(sentence)
        for key in keywords:
            res[key] = tfidf_index(terms=sentence.count(key),
                                   total_terms=len(sentence),
                                   docs=0 if key not in self.keyword_id_mapping else len(self.keyword_id_mapping[key]),
                                   total_docs=len(self.kb_raw_dic))
        return res

    def _load_mapping(self) -> dict:
        """
        不存在倒排索引时生成，并写入json
        """
        if os.path.exists(self.mapping_path):
            return read_json(self.mapping_path)
        else:
            print('generate mapping...')
            mapping = {}
            kb_dict = self.kb_cut_dic if self.use_cut else self.kb_raw_dic
            for count, key in enumerate(self.word_count_dic):
                mapping_ids = []
                for _id, words in kb_dict.items():
                    if key in words:
                        mapping_ids.append(_id)
                mapping[key] = mapping_ids
            write_json(file_path=self.mapping_path, data=mapping)
            return mapping

    def _load_predict(self) -> Predict:
        self.args = get_argparse().parse_args()
        self.args.device = device
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

    def cut_with_stopwords(self, sentence):
        """
        jieba分词并去掉停用词
        """
        res = []
        if self.use_search_cut:
            k_words_list = jieba.lcut_for_search(sentence)
        else:
            k_words_list = jieba.lcut(sentence)
        for cut_w in k_words_list:
            if cut_w not in self.stop_words:
                res.append(cut_w)
        return res

    def keywords_search(self, origin_sentence, keywords: [str], weight: float = 1.0) -> (dict, set):
        """从倒排索引中查找句子id并计算权重"""
        # e.g: key:weight {'1':0.3}
        if self.use_tfidf:
            word_weight_dic = self._get_sentence_tfidf(origin_sentence, keywords)
        else:
            word_weight_dic = self.word_weight_dic

        key_id_weight_dict = collections.defaultdict(float)
        oov_keys = set()
        for keyword in keywords:
            if keyword not in self.keyword_id_mapping:
                oov_keys.add(keyword)
                continue
            # keywords对应的句子id
            ids = self.keyword_id_mapping[keyword]
            for _id in ids:
                key_id_weight_dict[_id] += word_weight_dic[keyword] * weight
        return key_id_weight_dict, oov_keys

    def corpus_search(self, corpus_id_weight_dict: dict, exist_keys: [set]) -> dict:
        """
        从corpus中再抽取新的关键字
        :param corpus_id_weight_dict: corpus索引列表
        :param exist_keys: 已有的key，需去除
        """
        corpus_key_dict = collections.defaultdict(int)
        exist_keys = [key for keys in exist_keys for key in keys]  # 扁平化列表
        # 对corpus关键词频率计数
        for _id in corpus_id_weight_dict:
            for key in self.kb_cut_dic[_id]:
                if key in exist_keys or key in self.stop_words:
                    continue
                corpus_key_dict[key] += 1
        # 取新关键词前20个进行计算
        corpus_key_dict = get_dict_topk(corpus_key_dict, 20)
        new_kb_dict, _ = self.keywords_search(origin_sentence=None, keywords=corpus_key_dict.keys(), weight=1.0)
        return get_dict_topk(new_kb_dict, 100)

    def get_keywords(self, question: str, background: str) -> (set, set):
        keywords = self.predict(question, background)
        ques_keys = [item['text'] for item in keywords['in_question']]
        back_keys = [item['text'] for item in keywords['in_background']]
        return set(ques_keys), set(back_keys)

    def cal_similarity(self, qa_str: str, kb_dict: [str]) -> dict:
        """
        计算相似度
        :param qa_str: question:option
        :param kb_dict: 解析库语句id
        """
        qa_list = self.cut_with_stopwords(qa_str)
        new_dict = collections.defaultdict(float)
        for k in kb_dict:
            # 选择合适的相似度算法，并附加权重
            similarity = jaccard_similarity_set(qa_list, self.kb_cut_dic[k])
            new_dict[k] = similarity
        return new_dict


class Converter:
    def __init__(self, data_path: str, output_path, has_answer: bool, searcher: Searcher, knowledge_num=15):
        self.searcher = searcher
        self.data_path = data_path
        self.output_path = output_path
        self.has_answer = has_answer
        self.knowledge_num = knowledge_num
        self.processed_data = []
        self.corpus_limit = None
        self.options = ['A', 'B', 'C', 'D']
        self.weight = {'background': 0.2, 'question': 0.3, 'option': 0.5}

    def process_data(self):
        raw_data = read_json(self.data_path)
        print(f'question count: {len(raw_data)}')
        for ques_dic in raw_data:
            self.processed_data.append(self._get_ques_option(ques_dic))
        self._search_in_kb()
        write_json(self.output_path, self.processed_data)
        print('Process Done.')

    def _search_in_kb(self):
        for index, ques_dic in enumerate(self.processed_data):
            print(f'Searching {index}')
            # 1. question和background中查找
            # 利用模型抽取关键词
            ques_str, back_str = ques_dic['question'], ques_dic['background']
            ques_key, back_key = self.searcher.get_keywords(ques_str, back_str)
            # 关键词倒排检索
            ques_doc_id_dict, ques_oov = self.searcher.keywords_search(origin_sentence=ques_str,
                                                                       keywords=ques_key,
                                                                       weight=self.weight['question'])
            back_doc_id_dict, back_oov = self.searcher.keywords_search(origin_sentence=back_str,
                                                                       keywords=back_key,
                                                                       weight=self.weight['background'])
            # ques+back权重相加
            qb_kb_dict = merge_dict(ques_doc_id_dict, back_doc_id_dict, limit=self.corpus_limit)
            # 2. option中查找
            for option in self.options:
                qa_str = ques_dic[option]  # question:option
                option_str = qa_str.split(':')[-1]
                option_key = set(self.searcher.cut_with_stopwords(option_str))
                ques_dic[option] = {}
                # option关键词检索
                option_doc_id_dict, option_oov = self.searcher.keywords_search(origin_sentence=option_str,
                                                                               keywords=option_key,
                                                                               weight=self.weight['option'])
                # 与上一步权重相加
                kb_dict = merge_dict(option_doc_id_dict, qb_kb_dict, limit=self.knowledge_num * 200)
                # 权重归一
                norm_dict_value(kb_dict)
                # 3. 计算question+option与句子的相似度, 同时结果与kb_dict相加
                similarity_dict = self.searcher.cal_similarity(qa_str=qa_str, kb_dict=kb_dict)
                # 获取topK个，权重归一,得到第一步检索结果
                # ordered_kb = get_dict_topk(kb_dict, topk=self.knowledge_num*10)
                simi_kb_dict = merge_dict(kb_dict, similarity_dict, limit=self.knowledge_num * 10)
                norm_dict_value(simi_kb_dict)
                # 4. 分析第一步检索结果，抽取额外的关键词继续检索
                extend_kb_dict = self.searcher.corpus_search(simi_kb_dict, [ques_key, back_key, option_key])
                norm_dict_value(extend_kb_dict)
                # 5. 合并额外的知识
                res_kb_dict = merge_dict(simi_kb_dict, extend_kb_dict, weight1=0.8, weight2=0.2,
                                         limit=self.knowledge_num * 2)
                # 记录检索到的知识
                ques_dic[option][qa_str] = [self.searcher.kb_raw_dic[k] for k in res_kb_dict]
                # print(f"oov words: {[ques_oov, back_oov, option_oov]}")

    def _get_ques_option(self, ques_dic):
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


def generate_mapping_from_kb():
    counting = collections.defaultdict(int)
    mapping = collections.defaultdict(list)
    for _id, values in read_json('./data/all_kng_kb/kb_cut_use_geoV_kng.json').items():
        for v in set(values):
            # 去掉纯数字
            if len(v) > 1:  # and not v.isnumeric():
                counting[v] += 1
                mapping[v].append(_id)
    write_json(file_path='./data/mappings/kb_mapping.json', data=mapping)
    write_json(file_path='./data/all_kng_kb/kb_word_count.json', data=counting)
    print('Generating Done.')


def test():
    time_start = time.time()
    data_path = './data/test_data/beijingSimulation.json'
    # data_path = './data/raw/data_all/53_data.json'
    output_path = './output/search_result.json'
    # file_index = 'F'
    # data_path = f'./data/test_data/test_{file_index}.json'
    # output_path = f'output/search_result_{file_index}.json'
    searcher = Searcher(use_geo_vocabu=True, use_cut=True, use_tfidf=False)
    converter = Converter(data_path=data_path, output_path=output_path, has_answer=True, searcher=searcher,
                          knowledge_num=15)
    converter.process_data()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')


if __name__ == '__main__':
    device = '2'
    test()
    # generate_mapping_from_kb()
