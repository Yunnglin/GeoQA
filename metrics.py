from collections import defaultdict
from typing import List

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities, precision_recall_fscore_support

from raw_data_process import Word


def get_entity_from_labels(tokens: [str], labels: [str], detail=False, id2label=None) -> list:
    """
    extract entity from labels
    :param tokens: 输入序列
    :param labels: 输入标签
    :param id2label: 标签id转化
    :return: tokens中抽取的str序列
    """

    def get_token_words(_tokens, _labels):
        _res: [Word] = []
        for chunk in split_entity(_labels):
            start = chunk[1]
            end = chunk[2] + 1
            text = ''.join(_tokens[start:end])
            _res.append(Word(text, start, end))
        return _res

    if id2label:
        label_id_to_label(labels, id2label)
    if not detail:
        return get_token_words(tokens, labels)
    else:
        res = {'in_question': [], 'in_background': []}
        # 获取特殊标签位置
        cls_index = tokens.index('[CLS]')
        sep1_index = tokens.index('[SEP]')
        sep2_index = tokens.index('[SEP]', sep1_index + 1)
        # 分割tokens
        question_tokens = tokens[cls_index + 1:sep1_index]
        background_tokens = tokens[sep1_index + 1:sep2_index]
        # 分割labels
        question_labels = labels[0][cls_index + 1:sep1_index]
        background_labels = labels[0][sep1_index + 1:sep2_index]
        # 抽取结果
        res['in_question'] = [word.to_dict() for word in get_token_words(question_tokens, question_labels)]
        res['in_background'] = [word.to_dict() for word in get_token_words(background_tokens, background_labels)]
        return res


def split_entity(label_sequence):
    """
    从标签序列中抽取实体
        >>> label_sequence=[['O', 'B', 'O', 'B', 'I', 'B'], ['O', 'O', 'B']]
        >>> chunks=[('_', 1, 1), ('_', 3, 4), ('_', 5, 5), ('_', 9, 9)]
    :param label_sequence:
    :return: list of (chunk_type, chunk_start, chunk_end).
    """
    return get_entities(label_sequence)


def label_id_to_label(label_ids: List[List[int]], id2label: dict):
    """
    label_ids 转 label
    """
    for label_id in label_ids:
        for i, idx in enumerate(label_id):
            label_id[i] = id2label[idx]


def get_metrics(real_label, predict_label, id2label=None):
    if id2label:
        label_id_to_label(real_label, id2label)
        label_id_to_label(predict_label, id2label)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(real_label, predict_label)
    res_dict = {'precision': precision.item(),
                'recall': recall.item(),
                'f1_score': f_score.item(),
                'true_sum': true_sum.item()}
    return res_dict


def extract_tp_actual_correct(real_label, predict_label):
    # 获取对应的标签数量
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in split_entity(real_label):
        entities_true[type_name].add((start, end))
    for type_name, start, end in split_entity(predict_label):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return tp_sum, pred_sum, true_sum


def corrected_extract_metrics(tokens: List[List[str]], real_labels: List[List[str]], predict_labels: List[List[str]]):
    # 去掉位置信息，只需要抽取的实体
    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for token, real_label, predict_label in zip(tokens, real_labels, predict_labels):
        entities_true = set(get_entity_from_labels(token, real_label))
        entities_pred = set(get_entity_from_labels(token, predict_label))
        tp_sum = np.append(tp_sum, len(entities_true & entities_pred))
        pred_sum = np.append(pred_sum, len(entities_pred))
        true_sum = np.append(true_sum, len(entities_true))
    return tp_sum.sum(), pred_sum.sum(), true_sum.sum()


class Performance:
    def __init__(self, id2label=None):
        self.performance = {
            'tp_sum': 0,  # pred and true
            'pred_sum': 0,  # pred
            'true_sum': 0,  # true
        }
        self.id2label = id2label
        self.res_dict = {
            'precision': 0.0,  # tp_sum / pred_sum
            'recall': 0.0,  # tp_sum / true_sum
            'f1_score': 0.0
        }

    def update_performance(self, tokens: List[List[str]], real_labels, predict_labels):
        if self.id2label:
            label_id_to_label(real_labels, self.id2label)
            label_id_to_label(predict_labels, self.id2label)
        # performance_measure获取的是label级别的performance不正确，需要自行获取
        # 更新
        new_performance = corrected_extract_metrics(tokens, real_labels, predict_labels)
        self.performance['tp_sum'] += new_performance[0].item()
        self.performance['pred_sum'] += new_performance[1].item()
        self.performance['true_sum'] += new_performance[2].item()

    def _cal_performance(self):
        try:
            self.res_dict['precision'] = self.performance['tp_sum'] / self.performance['pred_sum']
            self.res_dict['recall'] = self.performance['tp_sum'] / self.performance['true_sum']
            self.res_dict['f1_score'] = 2 * self.res_dict['precision'] * self.res_dict['recall'] / (
                    self.res_dict['recall'] + self.res_dict['precision'])
            for key in self.res_dict:
                self.res_dict[key] = self.res_dict[key]
        except ZeroDivisionError as e:
            pass

    def __str__(self):
        self._cal_performance()
        info = " | ".join([f'{key}:{value:.4f}' for key, value in self.res_dict.items()])
        return info


if __name__ == '__main__':
    id2label = {0: 'O_LS_Y', 1: 'O', 2: 'B', 3: 'I'}
    y_true = [['O_LS_Y', 'O', 'O', 'B', 'I', 'B'], ['B', 'I', 'B']]
    y_pred = [['O_LS_Y', 'B', 'O', 'B', 'I', 'B'], ['O', 'O', 'B']]
    y_true_id = [[0, 1, 1, 2, 3, 2, ], [2, 3, 2]]
    y_pred_id = [[0, 1, 1, 2, 3, 2, ], [1, 1, 2]]
    p = Performance(id2label)
    p.update_performance([[0, 1, 1, 2, 3, 2, ]], [[0, 1, 1, 2, 3, 2, ]])
    print(p)
    p.res_dict['res'] = 2.1
    p.update_performance([[2, 3, 2]], [[1, 1, 2]])
    print(p)
    # print(split_entity(y_pred))
    y_true_id = [[0, 1, 1, 2, 3, 2, ], [2, 3, 2]]
    y_pred_id = [[0, 1, 1, 2, 3, 2, ], [1, 1, 2]]
    print(get_metrics(y_true_id, y_pred_id, id2label))
