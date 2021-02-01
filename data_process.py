import copy
import json


class InputExample(object):
    def __init__(self, guid, text_a, labels):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


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
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


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
                        datas_list.append(data_dict)
                        words = []
                        labels = []

                elif line == "" or line == '\n':
                    data_dict['text_a_len'] = text_a_len
                    text_a_len = 0
                else:
                    splits = line.split(' ')
                    words.append(splits[0])  # 词
                    labels.append(splits[1].replace('\n', ''))  # 词的标注
                    text_a_len += 1
            if words:
                datas_list.append(data_dict)
        return datas_list

print(DataProcessor.read_text('./data/train.txt')[0])