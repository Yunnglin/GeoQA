import os

import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

from config import get_argparse
from data_process import CnerProcessor as Processor, convert_examples_to_features


def load_and_cache_examples(args, tokenizer, data_type='train'):
    # 加载数据
    save_processed_data = './data/{}_processed_data'.format(data_type)
    if os.path.exists(save_processed_data):
        print('加载数据')
        features = torch.load(save_processed_data)
    else:
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                cls_token_at_end=False,
                                                pad_on_left=False,
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )

        # torch.save(features, save_processed_data)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


if __name__ == "__main__":
    args = get_argparse().parse_args()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.output_dir):  # 输出文件
        os.mkdir(args.output_dir)

    processor = Processor()
    label_list = processor.get_labels()
    # 将标签进行id映射
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained('./bert/vocab.txt')

    # 实例化模型
    # model = BertCrfForNer(num_labels)
    # model.to(device)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, data_type='train')
