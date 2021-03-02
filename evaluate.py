import logging
import os

import torch
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset
from transformers import BertTokenizer

from model import BertLSTMCRF
from model.bert_crf_model import BertCRF
from config import get_argparse
from data_process import collate_fn, CnerProcessor, convert_examples_to_features
from metrics import Performance


def load_and_cache_examples(args, tokenizer, processor, data_type='train'):
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

        torch.save(features, save_processed_data)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def load_model(args, num_labels, model_path):
    try:
        if 'lstm' in model_path:
            model = BertLSTMCRF(args, num_labels)
        else:
            model = BertCRF(args, num_labels)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError as e:
        print(e)
        print("Load Model Failed, No Such File")
        return None
    return model


def evaluate(args, model, tokenizer, processor, data_type):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() and args.device != '-1' else 'cpu')

    performance = Performance(id2label=args.id2label)
    eval_dataset = load_and_cache_examples(args, tokenizer, processor=processor, data_type=data_type)
    eval_sampler = SequentialSampler(eval_dataset)

    batch_size = args.eval_batch_size if data_type == "dev" else args.predict_batch_size
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size,
                                 collate_fn=collate_fn)

    logging.info("***** Running %s *****" % data_type)
    logging.info("  Num examples = %d" % len(eval_dataset))
    logging.info("  Batch size = %d" % args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'token_type_ids': batch[2], "labels": batch[3],
                      'input_lens': batch[4]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])

        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()  # 真实标签id
        tags = tags.squeeze(0).cpu().numpy().tolist()  # 预测标签id
        # 每个batch更新三项指标
        performance.update_performance(real_label=out_label_ids, predict_label=tags)

    # 输出指标
    logging.info("\n ***** %s results ***** " % data_type)
    eval_loss = eval_loss / nb_eval_steps
    performance.res_dict['eval_loss'] = eval_loss
    logging.info(performance)


if __name__ == "__main__":
    processor = CnerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    # 将标签进行id映射
    args = get_argparse().parse_args()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}

    model = load_model(args=args, num_labels=num_labels, model_path='./save_model/ckpt_epoch_9.bin')
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_path, 'vocab.txt'))

    evaluate(args, model, tokenizer, processor=processor, data_type="test")
