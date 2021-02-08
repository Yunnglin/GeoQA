from typing import List

from seqeval.metrics.sequence_labeling import get_entities, precision_recall_fscore_support


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
    for label_id in label_ids:
        for i, idx in enumerate(label_id):
            label_id[i] = id2label[idx]


def get_metrics(real_label, predict_label, id2label=None):
    if id2label:
        label_id_to_label(real_label, id2label)
        label_id_to_label(predict_label, id2label)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(real_label, predict_label)
    res_dict = {'precision': precision,
                'recall': recall,
                'f1_score': f_score,
                'true_sum': true_sum}
    return res_dict


if __name__ == '__main__':
    y_true = [['O_LS_Y', 'O', 'O', 'B', 'I', 'B'], ['B', 'I', 'B']]
    y_pred = [['O_LS_Y', 'B', 'O', 'B', 'I', 'B'], ['O', 'O', 'B']]
    print(split_entity(y_pred))
    print(get_metrics(y_true, y_pred))
