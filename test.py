import os
import uuid

from config import get_argparse
from predict import Predict
from utils.json_io import read_json, write_json


class Test:
    def __init__(self, test_data_path):
        self.test_data_path = test_data_path
        self.test_data = None
        self.predict = None

    def read_test_data(self):
        self.test_data = read_json(self.test_data_path)

    def run_test(self):
        print('running test')
        predict_res = []
        for data in self.test_data:
            data_id = data['id'] if 'id' in data else str(uuid.uuid3(uuid.NAMESPACE_DNS, data['question']))
            question = data['question']
            background = data['background']
            key_words = self.predict(question, background)
            predict_res.append({'id': data_id, 'keywords': key_words})
        return predict_res


class ModelInfo:
    def __init__(self, bert_type: str, name: str, path: str):
        self.bert_type = bert_type
        self.name = name
        self.path = path


if __name__ == '__main__':
    args = get_argparse().parse_args()
    args.max_seq_length = 256
    data_paths = ['/home/data_ti5_d/maoyl/nlp_scqa_geo/final_test_data/beijingSimulation.json']
    # data_paths = ['data/test_data/53_no_graph_test_data_95.json', 'data/test_data/53_graph_test_data_133.json']
    models = [ModelInfo('bert_base_chinese',
                        'bert_base_chinese-lstm-crf-cut-redundant-epoch_9.bin',
                        '/home/data_ti5_d/maoyl/GeoQA/save_model'),
              ModelInfo('bert_base_chinese',
                        'bert_base_chinese-lstm-crf-no_cut-redundant-epoch_9.bin',
                        '/home/data_ti5_d/maoyl/GeoQA/save_model'), ]
    for data_path in data_paths:
        test = Test(data_path)
        test.read_test_data()
        # test
        for model in models:
            args.bert_path = os.path.join('./bert', model.bert_type)

            model_path = os.path.join(model.path, model.name)
            print('model_path: ' + model_path)
            test.predict = Predict(args, model_path)
            res = test.run_test()
            # 写入测试结果
            save_name = data_path.split('/')[-1].split('.')[0]
            save_name = model.name.split('.')[0] + '-' + save_name
            write_json(os.path.join(args.result_path, save_name + '-keyword.json'),
                       res)
