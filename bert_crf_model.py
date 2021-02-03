from torch import nn
from transformers import BertConfig, BertModel

from crf import CRF


class BertCRF(nn.Module):
    def __init__(self, num_labels):
        super(BertCRF, self).__init__()
        # bert模型
        self.config = BertConfig.from_pretrained('./bert/bert_config.json')
        self.bert = BertModel.from_pretrained('./bert/pytorch_model.bin', config=self.config)

        # 每个token进行分类
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

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
