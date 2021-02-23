import os

import torch
from torch import nn
from transformers import BertConfig, BertModel

from model import CRF


class BertLSTMCRF(nn.Module):
    def __init__(self, args, num_labels):
        super().__init__()
        self.USE_CUDA = (args.device != '-1')
        self.output_size = num_labels
        self.rnn_layers = args.lstm_rnn_layers
        self.hidden_dim = args.lstm_hidden_dim
        self.bidirectional = args.lstm_bidirectional

        # bert模型，作为词嵌入
        self.bert_config = BertConfig.from_pretrained(os.path.join(args.bert_path, 'bert_config.json'))
        self.word_embeds = BertModel.from_pretrained(os.path.join(args.bert_path, 'pytorch_model.bin'),
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
        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.output_size)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.output_size)

        # crf layer
        self.crf = CRF(self.output_size)

    def init_hidden(self, batch_size):
        number = 2 if self.bidirectional else 1
        if self.USE_CUDA:
            hidden = (torch.randn(self.rnn_layers * number, batch_size, self.hidden_dim).cuda(),
                      torch.randn(self.rnn_layers * number, batch_size, self.hidden_dim).cuda())
        else:
            hidden = (torch.randn(self.rnn_layers * number, batch_size, self.hidden_dim),
                      torch.randn(self.rnn_layers * number, batch_size, self.hidden_dim))
        return hidden

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
        lstm_feats = self._get_lstm_features(sentence)

        # todo: use crf decode to get score
        # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion
