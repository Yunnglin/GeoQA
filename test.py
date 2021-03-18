import os

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
            data_id = hash(data['question'])
            question = data['question']
            background = data['background']
            key_words = self.predict(question, background)
            predict_res.append({'id': data_id, 'keywords': key_words})
        return predict_res


if __name__ == '__main__':
    args = get_argparse().parse_args()
    args.max_seq_length = 256
    data_paths = ['data/test_data/53_no_graph_test_data_95.json', 'data/test_data/53_graph_test_data_133.json']
    models = ['bert_base_chinese-lstm-crf-cut-redundant-epoch_14.bin',
              'bert_base_chinese-lstm-crf-no_cut-redundant-epoch_14.bin']

    for data_path in data_paths:
        test = Test(data_path)
        test.read_test_data()
        # test
        for root, dirs, model_names in os.walk('/home/maoyl/GeoQA/save_model/len_256_epoch_15_cut_redundant'):
            for model_name in model_names:
                if model_name not in models:
                    continue
                args.bert_path = os.path.join('./bert', model_name.split('-')[0])
                model_path = os.path.join(root, model_name)
                print('model_path: ' + model_path)
                test.predict = Predict(args, model_path)
                res = test.run_test()

                save_name = data_path.split('/')[-1].split('.')[0]
                save_name = 'no_cut-' + save_name if 'no_cut' in model_name else 'cut-' + save_name
                write_json(os.path.join(args.result_path, save_name + '-keyword.json'),
                           res)
