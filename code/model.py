import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import networkx as nx
from collections import defaultdict
import copy
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import BertForSequenceClassification

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea

    def run_eval(self, text):
        return self.encoder(text)[0]


class SimpleLSTMBaseline(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, emb_dim=64, num_linear=1):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc2 = nn.Linear(int(hidden_dim/2), 1)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        x = self.fc1(feature)
        preds = self.fc2(x)
        return preds.view(-1)

