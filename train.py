import argparse
import torch
from torch import nn
import numpy as np
from sample import Sample
from DataLoader import DataLoader
from model import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from util import save_weights, idx_to_sentence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Data = DataLoader()
Train_captions, Train_images, Train_feature, Train_lengths, img = Data.get_Training_data()
Test_captions, Test_images, Test_feature, Test_lengths = Data.get_val_data()


def main(args):
    encoder = Encoder(args.embed_size).to(device)
    decoder = Decoder(args.embed_size, args.hidden_size, len(Data.word_to_idx), args.num_layers).to(device)

    criterion = nn.CrossEntropyLoss()

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.rg)

    for epoch in range(args.num_epochs):
        avgTrainLoss = train_step(encoder, decoder, criterion, optimizer)
        avgTestLoss = val_step(encoder, decoder, criterion)

        print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
              .format(epoch + 1, args.num_epochs, avgTrainLoss,
                      np.exp(avgTrainLoss), avgTestLoss, np.exp(avgTestLoss)))

    save_weights(encoder, args.model_path + "encoder")

    save_weights(decoder, args.model_path + "decoder")

    sample = Sample(encoder, decoder, device)

    train_ouutput = sample.predict(Train_feature[Train_images[0:15]])
    test_ouutput = sample.predict(Test_feature[Test_images[0:15]])

    train_sent = idx_to_sentence(train_ouutput, Data.idx_to_word)
    train_GT = idx_to_sentence(Train_captions[0:15], Data.idx_to_word)

    test_sent = idx_to_sentence(test_ouutput, Data.idx_to_word)
    test_GT = idx_to_sentence(Test_captions[0:15], Data.idx_to_word)

    for i in range(15):
        print(train_sent[i])
        print(train_GT[i])
        print("")
        print("")

    print("")
    print("")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("")
    print("")

    for i in range(15):
        print(test_sent[i])
        print(test_GT[i])
        print("")
        print("")


def train_step(encoder, decoder, criterion, optimizer):
    Train_totalLoss = 0
    for i in range(int(Train_captions.shape[0] / args.batch_size)):
        start = args.batch_size * i

        Train_caption = Train_captions[start: start + args.batch_size]
        Train_image = Train_images[start: start + args.batch_size]
        Train_length = Train_lengths[start: start + args.batch_size]

        Train_fea = torch.from_numpy(Train_feature[Train_image])
        Train_image = torch.from_numpy(Train_image)
        Train_caption = torch.from_numpy(Train_caption)
        Train_length = torch.from_numpy(Train_length)

        Train_caption = Train_caption.long()
        Train_caption = Train_caption.to(device)
        Train_fea = Train_fea.to(device)

        Train_fea = encoder(Train_fea)

        targets = pack_padded_sequence(Train_caption, Train_length, batch_first=True, enforce_sorted=False)[0]

        outputs = decoder(Train_fea, Train_caption, Train_length)

        loss = criterion(outputs, targets)
        Train_totalLoss += loss.item() * args.batch_size

        decoder.zero_grad()
        encoder.zero_grad()

        loss.backward()

        optimizer.step()
    return Train_totalLoss / Train_captions.shape[0]


def val_step(encoder, decoder, criterion):
    Test_totalLoss = 0

    for i in range(int(Test_captions.shape[0] / args.batch_size)):
        start = args.batch_size * i
        Test_caption = Test_captions[start: start + args.batch_size]
        Test_image = Test_images[start: start + args.batch_size]
        Test_length = Test_lengths[start: start + args.batch_size]

        Test_fea = torch.from_numpy(Test_feature[Test_image])
        Test_image = torch.from_numpy(Test_image)
        Test_caption = torch.from_numpy(Test_caption)
        Test_length = torch.from_numpy(Test_length)

        Test_caption = Test_caption.long()
        Test_caption = Test_caption.to(device)
        Test_fea = Test_fea.to(device)

        Test_fea = encoder(Test_fea)
        targets = pack_padded_sequence(Test_caption, Test_length, batch_first=True, enforce_sorted=False)[0]

        outputs = decoder(Test_fea, Test_caption, Test_length)
        # outputs = outputs.reshape(int(outputs.size(0) / 17), 1004, 17)

        loss = criterion(outputs, targets)
        Test_totalLoss += loss.item() * args.batch_size

    return Test_totalLoss / Test_captions.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--rg', type=float, default=0)
    args = parser.parse_args()

    print(args)
    main(args)
