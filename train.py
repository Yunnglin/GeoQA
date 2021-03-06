import logging
import os
import time

import torch
from torch.utils.data import RandomSampler, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from model import BertLSTMCRF, BertCRF
from config import get_argparse
from data_process import CnerProcessor, collate_fn
from evaluate import evaluate, load_and_cache_examples
from utils.logger import setup_logging


def train(args):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() and args.device != '-1' else 'cpu')

    processor = CnerProcessor()
    label_list = processor.get_labels()
    # 将标签进行id映射
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # 模型名称
    store_name = args.bert_path.split('/')[-1]
    # 实例化模型
    if args.use_lstm:
        model = BertLSTMCRF(args=args, num_labels=num_labels)
        store_name += '-lstm-crf'
    else:
        model = BertCRF(args=args, num_labels=num_labels)
        store_name += '-crf'
    model.to(device)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, processor=processor, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collate_fn)
        t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epochs

        no_decay = ["bias", "LayerNorm.weight"]
        if not args.use_lstm:
            bert_param_optimizer = list(model.bert.named_parameters())
            linear_param_optimizer = list(model.classifier.named_parameters())
        else:
            bert_param_optimizer = list(model.word_embeds.named_parameters())
            linear_param_optimizer = list(model.hidden2tag.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate}
        ]

        # 定义优化器
        warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        logging.info("\tRunning training")
        logging.info("\tNum examples = %d" % len(train_dataset))
        logging.info("\tNum Epochs = %d" % args.epochs)
        logging.info("\tGradient Accumulation steps = %d" % args.gradient_accumulation_steps)
        logging.info("\tTotal optimization steps = %d" % t_total)
        logging.info("\tModel type %s" % store_name)
        global_step = 0
        model.zero_grad()
        for epoch in range(int(args.epochs)):
            model.train()
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2],
                          'labels': batch[3], 'input_lens': batch[4]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                print('epoch:{}, step:{}, loss:{:10f}, time:{:10f}'.format(epoch, step, loss, time.time() - start_time))

                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

            # evaluate效果
            logging.info(f'train_epoch: {epoch}')
            performance = evaluate(args, model, tokenizer, processor=processor, data_type="dev")
            logging.info("eval_performance:\t" + str(performance))

            # 每训练5轮保存一次
            if epoch > 5 and (epoch + 1) % 5 == 0:
                model_path = os.path.join(args.checkpoint_path, f"{store_name}-epoch_{epoch}.bin")
                save_model(model=model, model_path=model_path)


def save_model(model, model_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    # 保存参数
    torch.save(model_to_save.state_dict(), model_path)
    logging.info("Saved model at" + model_path)


def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
    """
    model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


if __name__ == "__main__":
    setup_logging(default_path='./utils/logger_config.json')
    args = get_argparse().parse_args()

    if not os.path.exists(args.checkpoint_path):  # 模型保存路径
        os.mkdir(args.checkpoint_path)

    bert_names = ['bert_base_chinese', 'albert_chinese_large', 'chinese_bert_wwm_ext', 'chinese_roberta_wwm_ext_large']
    args.epochs = 20

    for name in bert_names:
        args.bert_path = './bert/' + name
        # bert_crf
        args.use_lstm = False
        train(args=args)
        # bert_lstm_crf
        args.use_lstm = True
        train(args=args)

    # args.bert_path = './bert/' + bert_names[3]
    # train(args=args)
