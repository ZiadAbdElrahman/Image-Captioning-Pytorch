import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()

        # self.bn1 = nn.BatchNorm1d(512,  momentum=0.01)
        self.linear1 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024,  momentum=0.01)
        self.linear2 = nn.Linear(1024, embed_size)
        self.bn3 = nn.BatchNorm1d(embed_size,  momentum=0.01)

    def forward(self, image):

        # image = (self.bn1(image))
        feature = (self.linear1(image))
        feature = (self.bn2(feature))
        feature = (self.linear2(feature))
        feature = (self.bn3(feature))

        return feature



class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(1004, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, feature, captions, lengths):

        embeddings = self.embed(captions)
        embeddings = torch.cat((feature.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        hiddens, _ = self.lstm(packed)
        output = self.linear1(hiddens[0])


        return output

    def sample(self, feature, states):
        sampled_ids = []

        inputs = feature.unsqueeze(1)

        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear1(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


