import os

import torch
from torch import nn
from transformers import BertConfig, BertModel, AutoConfig, AutoModel

from model import CRF


class BertLSTMCRF(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion
    """

    def __init__(self, args, num_labels):
        super().__init__()
        self.device = torch.device(
            'cuda:{}'.format(args.device) if torch.cuda.is_available() and args.device != '-1' else 'cpu')
        self.output_size = num_labels
        self.rnn_layers = args.lstm_rnn_layers
        self.hidden_dim = args.lstm_hidden_dim
        self.bidirectional = args.lstm_bidirectional

        # bert模型，作为词嵌入
        self.bert_config = AutoConfig.from_pretrained(os.path.join(args.bert_path, 'config.json'))
        self.word_embeds = AutoModel.from_pretrained(os.path.join(args.bert_path, 'pytorch_model.bin'),
                                                     config=self.bert_config)

        # lstm layers
        self.lstm = nn.LSTM(input_size=self.bert_config.hidden_size,
                            # input_size: The number of expected features in the input `x`
                            hidden_size=self.hidden_dim,  # The number of features in the hidden state `h`
                            num_layers=self.rnn_layers,
                            # Number of recurrent layers. E.g., setting ``num_layers=2``would mean stacking two LSTMs
                            # together to form a `stacked LSTM`,with the second LSTM taking in outputs of the first
                            # LSTM and computing the final results. Default: 1
                            batch_first=True,
                            bidirectional=self.bidirectional)  # If True, becomes a bidirectional LSTM.
        # dropout layer
        self.dropout = nn.Dropout(args.lstm_dropout)

        # linear layer
        # Maps the output of the LSTM into tag space.
        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.output_size)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.output_size)

        # crf layer
        self.crf = CRF(num_tags=self.output_size, batch_first=True)

    def init_hidden(self, batch_size):
        number = 2 if self.bidirectional else 1

        return (torch.randn(self.rnn_layers * number, batch_size, self.hidden_dim).to(self.device),
                torch.randn(self.rnn_layers * number, batch_size, self.hidden_dim).to(self.device))

    def _get_lstm_features(self, sentence):
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(sentence, hidden)
        # contiguous 使其变得内存相邻
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        d_lstm_out = self.dropout(lstm_out)
        l_out = self.hidden2tag(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        emdeds = self.word_embeds(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        sentence = emdeds[0]  # shape=(Batch Size, Input len, Hidden size)
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # print(lstm_feats.shape) => (batch_size, seq_length, num_tags)

        # use crf to get loss
        outputs = (lstm_feats,)
        if labels is not None:
            # Compute the conditional log likelihood of a sequence of tags given emission scores
            loss = self.crf(emissions=lstm_feats, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss, emission scores)
