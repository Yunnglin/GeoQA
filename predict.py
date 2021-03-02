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


class Predict:

    def __init__(self, args, model_path):
        import warnings

        warnings.filterwarnings("ignore")
        args.device = '-1'

        processor = CnerProcessor()
        label_list = processor.get_labels()
        num_labels = len(label_list)

        args.id2label = {i: label for i, label in enumerate(label_list)}
        args.label2id = {label: i for i, label in enumerate(label_list)}

        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.model = load_model(args=args, num_labels=num_labels,
                                model_path=model_path)

    def predict(self, question, background):
        question_l = process_raw_sentence(question)
        background_l = process_raw_sentence(background)
        # print(question_l)
        # print(background_l)

        # tokens, input_ids, input_mask, segment_ids, input_len
        data_tuple = convert_raw_sentence_to_features(question=question_l,
                                                      background=background_l,
                                                      tokenizer=self.tokenizer)
        with torch.no_grad():
            input_ids = torch.LongTensor([data_tuple[1]])
            input_mask = torch.LongTensor([data_tuple[2]])
            segment_ids = torch.LongTensor([data_tuple[3]])
            input_len = torch.LongTensor([data_tuple[4]])
            inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids,
                      'input_lens': input_len}
            outputs = self.model(**inputs)
            emissions = outputs[0]

            tags = self.model.crf.decode(emissions, inputs['attention_mask'])
            # print(tags.size()) [1,1,max_seq_len]
            tags = tags.squeeze(0).cpu().numpy().tolist()

            res = {'predict_entity': get_entity_from_labels(tokens=data_tuple[0], labels=tags,
                                                            id2label=self.args.id2label),
                   'tag_seq': tags}
            return res

    def __call__(self, question, background):
        print(self.predict(question, background))


if __name__ == '__main__':
    question = "我国下列地区中，资源条件最适宜建太阳能光热电站的是"
    background = "考点二太阳对地球的影响\n太阳能光热电站（下图）通过数以十万计的反光板聚焦太阳 能，给高塔顶端的锅炉加热，产生蒸汽，驱动发电机发电。据此 完成下题。"

    predict = Predict(args=get_argparse().parse_args(),
                      model_path='./save_model/albert_chinese_large_lstm_crf_epoch_0.bin')
    predict(question, background)

    question = "欧盟进口的玩具80%来自我国，主要是由于我国玩具"
    scenario_text = "2013年7月 20 日起《欧盟新玩具安全指令》正式实施，在物理、化学、电子、卫生、辐射等诸多领域里做出了“世界上最严格的规定”。读下图，完成下列各题。"
    predict(question, scenario_text)
