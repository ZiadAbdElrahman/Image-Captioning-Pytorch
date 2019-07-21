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


class Attention(nn.Module):

    def __init__(self, decoder_dim, encoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.embd_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, attention_dim, vocab_size, max_seq_length=17):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(1004, embed_size)
        self.attention = Attention(hidden_size, embed_size, attention_dim)  # attention network
        self.lstm = nn.LSTMCell(2*embed_size, hidden_size)
        self.f_beta = nn.Linear(hidden_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.max_seg_length = max_seq_length
        self.init_h = nn.Linear(hidden_size, embed_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(hidden_size, embed_size)  # linear layer to find initial cell state of LSTMCell

    def init_hidden_state(self, encoder_out):

        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, feature, captions, length):
        outputs = []
        # h = feature
        # c = torch.zeros_like(feature)
        h, c = self.init_hidden_state(feature)  # (batch_size, decoder_dim)
        embeddings = self.embed(captions)
        num_of_wards = int(length[0])
        num_of_wards = 16

        for i in range(num_of_wards):
            attention_weighted_encoding, _ = self.attention(feature, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            inputs = torch.cat([attention_weighted_encoding, embeddings[:, i, :]], dim=1)
            # inputs = embeddings[:, i, :]
            h, c = self.lstm(inputs, (h, c))
            outputs.append(h)

        outputs = torch.stack(outputs, 1)
        outputs = self.linear(outputs)
        return outputs

    def sample(self, feature, inputs):
        sampled_ids = []
        h = feature
        c = torch.zeros_like(feature)
        inputs = self.embed(inputs)
        h, c = self.init_hidden_state(feature)  # (batch_size, decoder_dim)

        for i in range(self.max_seg_length):
            # attention_weighted_encoding, _ = self.attention(feature, h)
            # gate = self.sigmoid(self.f_beta(h))
            # attention_weighted_encoding = gate * attention_weighted_encoding
            # inputs = torch.cat([attention_weighted_encoding, inputs], dim=1)
            h, c = self.lstm(inputs, (h, c))
            outputs = self.linear(h)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
