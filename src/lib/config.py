from dataclasses import dataclass
import torch


@dataclass
class Variables:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    TARGETS = {
        0: 'trebuet_remonta',
        1: 'kosmeticheskii',
        2: 'evro',
        3: 'dizainerskii'
    }
    EMPTY_TOKEN = 0
    NOT_IN_VOCAB = 1
    LSTM_CONCAT_PARAMS = {'vocab_size': 5584 + 2, #len(encode_mapping)=5584
        'embedding_dim': 256,
        'hidden_dim': 256,
        'n_layers': 1,
        'use_bidirectional': True,
        'dropout': 0.1,
        'pretrained': True
    }
