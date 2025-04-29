import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

class LSTM_NLP(nn.Module):
    def __init__(self):
        super(LSTM_NLP, self).__init__()

        def lstm_block(input_size, hidden_size, num_layers, bidirectional):
            out, (hn,cn) = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=0.24)

