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

        self.linear = nn.Linear(512, embed_size)
        self.bn = nn.BatchNorm1d(embed_size,  momentum=0.01)

    def forward(self, image):
        # with torch.no_grad():
        #     feature = self.resnet(image)

        # feature = feature.reshape(feature.size(0), -1)
        feature = (self.linear(image))

        return feature



class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(1004, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, feature, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((feature.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        hiddens, _ = self.lstm(packed)

        output = self.linear(hiddens[0])

        return output

    def sample(self, feature, start):
        sampled_ids = []


        states = start
        inputs = feature.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)


        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

