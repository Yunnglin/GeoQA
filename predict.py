import torch
from transformers import BertTokenizer

from config import get_argparse
from data_process import CnerProcessor
from evaluate import load_model
from metrics import get_entity_from_labels
from raw_data_process import preprocess


def convert_raw_sentence_to_features(question,
                                     background,
                                     tokenizer,
                                     max_seq_length=128,
                                     cls_token_at_end=False,
                                     cls_token="[CLS]",
                                     cls_token_segment_id=0,
                                     sep_token="[SEP]",
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     mask_padding_with_zero=True,
                                     ):
    tokens = question + background
    text_a_len = len(question)

    if len(tokens) > max_seq_length - 3:  # CLS + SEP + SEP
        tokens = tokens[: (max_seq_length - 3)]
    text_a_len = len(tokens) if text_a_len > len(tokens) else text_a_len

    # 添加SEP标记,
    tokens += [sep_token]
    tokens.insert(text_a_len, sep_token)

    text_a_len += 1  # 扩展一位SEP
    segment_ids = [sequence_a_segment_id] * text_a_len
    segment_ids += [sequence_b_segment_id] * (len(tokens) - text_a_len)

    # 添加CLS标记
    tokens = [cls_token] + tokens
    # 判断是否有重合词
    segment_ids = [cls_token_segment_id] + segment_ids

    # token转id
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    input_len = len(input_ids)
    padding_length = max_seq_length - input_len
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
    try:
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
    except AssertionError:
        print("ERROR: ")
        raise Exception
    finally:
        print("tokens: {}".format(" ".join([str(x) for x in tokens])))
        print("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
        print("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
        print("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))
        return tokens, input_ids, input_mask, segment_ids, input_len


def process_raw_sentence(sentence: str) -> [str]:
    return list(preprocess(sentence))


def predict(args, question, background, model, tokenizer):
    question_l = process_raw_sentence(question)
    background_l = process_raw_sentence(background)
    # print(question_l)
    # print(background_l)

    # tokens, input_ids, input_mask, segment_ids, input_len
    data_tuple = convert_raw_sentence_to_features(question=question_l,
                                                  background=background_l,
                                                  tokenizer=tokenizer)
    with torch.no_grad():
        input_ids = torch.LongTensor([data_tuple[1]])
        input_mask = torch.LongTensor([data_tuple[2]])
        segment_ids = torch.LongTensor([data_tuple[3]])
        input_len = torch.LongTensor([data_tuple[4]])
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids,
                  'input_lens': input_len}
        outputs = model(**inputs)
        emissions = outputs[0]

        tags = model.crf.decode(emissions, inputs['attention_mask'])
        # print(tags.size()) [1,1,max_seq_len]
        tags = tags.squeeze(0).cpu().numpy().tolist()

        res = {}
        res['extract_entity'] = get_entity_from_labels(tokens=data_tuple[0], labels=tags, id2label=args.id2label)
        res['tag_seq'] = tags
        return res


def run_predict(question, background):
    # 将标签进行id映射
    args = get_argparse().parse_args()
    args.device = '-1'

    processor = CnerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}

    tokenizer = BertTokenizer.from_pretrained('./bert/vocab.txt')
    model = load_model(args=args, num_labels=num_labels, model_path='./save_model/ckpt_lstm_epoch_9.bin')

    res = predict(args=args,
                  question=question,
                  background=background,
                  model=model,
                  tokenizer=tokenizer)

    print(res)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    question = "我国下列地区中，资源条件最适宜建太阳能光热电站的是"
    background = "考点二太阳对地球的影响\n太阳能光热电站（下图）通过数以十万计的反光板聚焦太阳 能，给高塔顶端的锅炉加热，产生蒸汽，驱动发电机发电。据此 完成下题。"
    run_predict(question, background)
