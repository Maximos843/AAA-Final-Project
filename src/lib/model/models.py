import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F
from src.lib.config import Variables
import pickle


class ConcatModelLSTM(nn.Module):
    def __init__(self, vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            n_layers: int = 1,
            use_bidirectional: bool = True,
            dropout: float = 0.1,
            pretrained: bool = True
        ):

        super(ConcatModelLSTM, self).__init__()
        self.resnet = nn.Sequential(*(list(resnet50(pretrained=pretrained).children())[:-1]))
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim // 2,
                           num_layers=n_layers,
                           batch_first=True,
                           bidirectional=use_bidirectional,
                           dropout=dropout)

        self.after_linear1 = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.after_linear2 = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.linear2 = nn.Linear(hidden_dim, 16)
        self.linear1 = nn.Linear(2048, 16)
        self.linear3 = nn.Linear(32, 4)

    def attention(self, lstm_output, final_state):
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.unsqueeze(2) #.squeeze(0) - убрал, тк батч из 1 элемента и по размерам краш
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x,  text):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        attn_output = self.attention(output, hidden)#.squeeze(0) аналогичная проблема, что и выше, поэтому убрал
        x = self.after_linear1(self.linear1(x))
        attn_output = self.after_linear2(self.linear2(attn_output.float()))
        x = torch.cat((x, attn_output), dim=-1)
        return self.linear3(x)


def create_model() -> ConcatModelLSTM:
    model = ConcatModelLSTM(**Variables.LSTM_CONCAT_PARAMS).to(Variables.DEVICE)
    model.load_state_dict(torch.load('/app/src/lib/model/best_model.pth', map_location=Variables.DEVICE))
    return model


def create_vectorizer() -> dict:
    with open('/app/src/lib/model/vectorizer.pkl', 'rb') as file:
        tokenizer, encode_mapping = pickle.load(file)
    return {'tokenizer': tokenizer, 'encode_mapping': encode_mapping}


'''class ResNet(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        self.resnet = nn.Sequential(*(list(resnet50(pretrained=pretrained).children())[:-1]))
        self.linear1 = nn.Linear(2048, 4)
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, (2 / 2048) ** 0.5)

    def freeze_weights(self):
        for i, param in enumerate(self.resnet.parameters()):
            if i > 100:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x'''