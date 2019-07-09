import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()

        # resnet = models.resnet152(pretrained=True)
        #
        # model = list(resnet.children())[:-1]
        #
        # self.resnet = nn.sequential(*model)
        self.linear1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024,  momentum=0.01)
        self.linear2 = nn.Linear(1024, embed_size)
        self.bn2 = nn.BatchNorm1d(embed_size,  momentum=0.01)

    def forward(self, image):
        # with torch.no_grad():
        #     feature = self.resnet(image)

        # feature = feature.reshape(feature.size(0), -1)

        feature = (self.linear1(image))
        feature = (self.bn1(feature))
        feature = (self.linear2(feature))
        feature = (self.bn2(feature))

        return feature



class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(1004, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 512)
        self.bn = nn.BatchNorm1d(512,  momentum=0.01)
        self.linear2 = nn.Linear(512, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, feature, captions, lengths):

        embeddings = self.embed(captions)
        embeddings = torch.cat((feature.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        hiddens, _ = self.lstm(packed)

        output = self.linear1(hiddens[0])
        output = self.bn(output)
        output = self.linear2(output)

        return output

    def sample(self, feature, states):
        sampled_ids = []

        inputs = feature.unsqueeze(1)

        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear1(hiddens.squeeze(1))
            outputs = self.bn(outputs)
            outputs = self.linear2(outputs)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

