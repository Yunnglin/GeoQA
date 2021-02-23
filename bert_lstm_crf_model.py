from torch import nn


class BertLSTMCRF(nn.Module):
    def __init__(self):
        super().__init__()
