import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()

        # self.bn1 = nn.BatchNorm1d(512,  momentum=0.01)
        self.linear1 = nn.Linear(512, 512)
        # self.bn2 = nn.BatchNorm1d(1024,  momentum=0.01)
        # self.linear2 = nn.Linear(1024, embed_size)
        # self.bn3 = nn.BatchNorm1d(embed_size)
        # self.apply(weights_init_normal)

    def forward(self, image):
        # image = (self.bn1(image))
        feature = (self.linear1(image))
        # feature = (self.bn2(feature))
        # feature = (self.linear2(feature))
        # feature = (self.bn3(feature))
        # feature = image
        return feature


def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(Decoder, self).__init__()
        self.func = torch.nn.Softmax(dim=2)
        self.embed = nn.Embedding(1004, embed_size)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length


        # self.apply(weights_init_normal)

    def forward(self, feature, captions, lengths):
        # packed = pack_padded_sequence(captions, lengths, batch_first=True)
        # print(packed[0])
        # print(captions[:, 0])
        embeddings = self.embed(captions)
        # print(embeddings.shape, feature.shape)
        # start = []
        # start.append()
        h = feature
        c = torch.zeros_like(feature)

        # embeddings = torch.cat((feature.unsqueeze(1), embeddings), 1)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # print(packed[0].shape)
        outputs = []
        # start = fstart
        for i in range(16):
            inputs = embeddings[:, i, :]
            h, c = self.lstm(inputs, (h, c))
            output = self.linear1(h)
            outputs.append(output)
            # _, predicted = output.max(1)
            # inputs = self.embed(predicted).unsqueeze(1)
            # h = start[0] + fstart[0]
            # c = start[1] + fstart[1]
            # start = []
            # start.append(h)
            # start.append(c)
        outputs = torch.stack(outputs, 1)
        return (outputs)

    def sample(self, feature, states, d):
        sampled_ids = []

        states = []
        h = feature
        c = torch.zeros_like(feature)
        # states = start

        inputs = self.embed(torch.ones(feature.shape[0]).long().to(d))
        for i in range(16):
            h, c = self.lstm(inputs, (h, c))
            outputs = self.linear1(h)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            # h = states[0] + start[0]
            # c = states[1] + start[1]
            # states = []
            # states.append(h)
            # states.append(c)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
