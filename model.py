import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, image):
        feature = self.bn1(self.linear1(image))
        return feature


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, max_seq_length=17):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(1004, embed_size)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, feature, captions, lengths):
        outputs = []
        h = feature
        c = torch.zeros_like(feature)
        embeddings = self.embed(captions)
        num_of_wards = embeddings.shape[1] - 1

        for i in range(num_of_wards):
            h, c = self.lstm(embeddings[:, i, :], (h, c))
            outputs.append(h)
            h += feature

        outputs = torch.stack(outputs, 1)
        outputs = self.linear(outputs)
        return outputs

    def sample(self, feature, inputs):
        sampled_ids = []
        h = feature
        c = torch.zeros_like(feature)
        inputs = self.embed(inputs)

        for i in range(self.max_seg_length):
            h, c = self.lstm(inputs, (h, c))
            outputs = self.linear(h)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            h += feature
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
