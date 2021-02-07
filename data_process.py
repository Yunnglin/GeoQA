import copy
import json
import os

import torch


def collate_fn(batch):
    """
    将batch根据最长的样例截断
    map(function, iterable)
    `batch` is a list of tuple id mask token_type_id len label
    stack 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
    在自然语言处理和卷及神经网络中， 通常为了保留–[序列(先后)信息] 和 [张量的矩阵信息] 才会使用stack。
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


class InputExample(object):
    def __init__(self, guid, text, text_a_len, labels):
        self.guid = guid
        self.text = text
        self.text_a_len = text_a_len
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def read_text(cls, input_file):
        datas_list = []
        words = []
        labels = []
        text_a_len = 0  # 句子A的长度
        data_dict = {'words': words, 'labels': labels, 'text_a_len': text_a_len}
        with open(input_file, 'r', encoding='utf8') as f:
            for line in f:
                if line.startswith('-DOC_START-'):
                    if words:
                        data_dict['words'] = words
                        data_dict['labels'] = labels
                        datas_list.append(data_dict.copy())
                        words = []
                        labels = []
                        text_a_len = 0
                elif line == "" or line == '\n':
                    data_dict['text_a_len'] = text_a_len
                else:
                    try:
                        splits = line.strip().split(' ')
                        words.append(splits[0])  # 词
                        labels.append(splits[1])  # 词的标注
                        text_a_len += 1
                    except IndexError:
                        print(splits[0])
            if words:
                data_dict['words'] = words
                data_dict['labels'] = labels
                datas_list.append(data_dict.copy())
        return datas_list


class CnerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self.read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self.read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """
        常规BIO标签
        <CLS_Y> 该数据有重合词时CLS的标签
        <CLS_N> 该数据没有重合词时CLS的标签
        <SEP> SEP的标签
        <PAD> PAD的标签
        """
        return ["<PAD>", "B", "I", "O", "<CLS_Y>", "<CLS_N>", "<SEP>", ]

    def _create_examples(self, lines, set_type):
        #  {"words": [词1, 词2....],"text_a_len":int, 'labels":[label1, label2, ...]}
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line['words']
            text_a_len = line['text_a_len']
            labels = line['labels']
            examples.append(InputExample(guid=guid, text=text, text_a_len=text_a_len, labels=labels))
        return examples


def convert_examples_to_features(examples: [InputExample],
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True, ):
    label_map = {label: i for i, label in enumerate(label_list)}
    map_label = {i: label for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = example.text
        text_a_len = example.text_a_len
        label_ids = [label_map[x] for x in example.labels]

        if len(tokens) > max_seq_length - 3:  # CLS + SEP + SEP
            tokens = tokens[: (max_seq_length - 3)]
            label_ids = label_ids[: (max_seq_length - 3)]
        text_a_len = len(tokens) if text_a_len > len(tokens) else text_a_len

        # 添加SEP标记,
        tokens += [sep_token]
        tokens.insert(text_a_len, sep_token)

        label_ids += [label_map['<SEP>']]
        label_ids.insert(text_a_len, label_map['<SEP>'])

        text_a_len += 1  # 扩展一位SEP
        segment_ids = [sequence_a_segment_id] * text_a_len
        segment_ids += [sequence_b_segment_id] * (len(tokens) - text_a_len)

        # 添加CLS标记
        tokens = [cls_token] + tokens
        # 判断是否有重合词
        label_ids = [label_map['<CLS_Y>'] if label_map['B'] in label_ids else label_map['<CLS_N>']] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        # token转id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        padding_length = max_seq_length - input_len
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([label_map['<PAD>']] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [label_map['<PAD>']] * padding_length
        try:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
        except AssertionError:
            print("ERROR: ")
            print("tokens: {}".format(" ".join([str(x) for x in tokens])))
            print("labels: {}".format(" ".join([str(map_label[x]) for x in label_ids])))
            print("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
            print("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
            print("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))
            print("label_ids: {}".format(" ".join([str(x) for x in label_ids])))
            raise Exception

        if ex_index < 5:
            print("*** Example ***")
            print("guid: {}".format(example.guid))
            print("tokens: {}".format(" ".join([str(x) for x in tokens])))
            print("labels: {}".format(" ".join([str(map_label[x]) for x in label_ids])))
            print("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
            print("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
            print("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))
            print("label_ids: {}".format(" ".join([str(x) for x in label_ids])))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))

    return features


if __name__ == '__main__':
    # content = DataProcessor.read_text('./data/train.txt')
    process = CnerProcessor()
    # examples = process.get_train_examples('./data')
    # for e in examples:
    #     print(e.to_json_string())
