import os

from torch import nn
from transformers import BertConfig, BertModel

from model import CRF


class BertCRF(nn.Module):
    def __init__(self, args, num_labels):
        super(BertCRF, self).__init__()
        # bert模型
        self.bert_config = BertConfig.from_pretrained(os.path.join(args.bert_path, 'bert_config.json'))
        self.bert = BertModel.from_pretrained(os.path.join(args.bert_path, 'pytorch_model.bin'),
                                              config=self.bert_config)

        # 每个token进行分类
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert_config.hidden_size, num_labels)

        # 送入CRF进行预测
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # shape=(Batch Size, Input len, Hidden size)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # shape=(Batch Size, Input len, Label size)

        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores
