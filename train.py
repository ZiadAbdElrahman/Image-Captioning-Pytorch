import torch
import random
import argparse
import numpy as np
from torch import nn
from sample import Sample
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from model import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from util import save_weights, idx_to_sentence, load_weights, image_from_url

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Data = DataLoader()
Train_captions, Train_images, Train_feature, Train_lengths = Data.get_Training_data()
Test_captions, Test_images, Test_feature, Test_lengths = Data.get_val_data()


def main(args):
    # writer = SummaryWriter()

    encoder = Encoder(args.embed_size).to(device)
    decoder = Decoder(args.embed_size, args.hidden_size, len(Data.word_to_idx), args.num_layers).to(device)

    # load_weights(encoder, args.model_path + "2.04encoder")
    # load_weights(decoder, args.model_path + "2.04decoder")

    criterion = nn.CrossEntropyLoss()

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.reg)

    for epoch in range(args.num_epochs):
        avgTrainLoss = train_step(encoder, decoder, criterion, optimizer)
        avgTestLoss = val_step(encoder, decoder, criterion)

        print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
              .format(epoch + 1, args.num_epochs, avgTrainLoss,
                      np.exp(avgTrainLoss), avgTestLoss, np.exp(avgTestLoss)))

    save_weights(encoder, args.model_path + "encoder")
    save_weights(decoder, args.model_path + "decoder")
    #
    # load_weights(encoder, args.model_path + "encoder")
    # load_weights(decoder, args.model_path + "decoder")

    sample = Sample(encoder, decoder, device)

    Train_mask = []
    for i in range(5):
        Train_mask.append(random.randint(0, Train_captions.shape[0]))
    Test_mask = []
    for i in range(30):
        Test_mask.append(random.randint(0, Test_captions.shape[0]))

    train_ouutput = sample.predict(Train_feature[Train_images[Train_mask]])
    test_ouutput = sample.predict(Test_feature[Test_images[Test_mask]])

    train_sent = idx_to_sentence(train_ouutput, Data.idx_to_word)
    train_GT = idx_to_sentence(Train_captions[Train_mask], Data.idx_to_word)

    test_sent = idx_to_sentence(test_ouutput, Data.idx_to_word)
    test_GT = idx_to_sentence(Test_captions[Test_mask], Data.idx_to_word)

    for i in range(5):
        print(train_sent[i])
        print(train_GT[i])
        print("")
        try:
            img = image_from_url(Data.train_urls[Train_images[Train_mask[i]]])
            plt.imshow(img)
            plt.show()
        except:
            print("error")

    print("")
    print("")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("")
    print("")

    for i in range(30):
        print(test_sent[i])
        print(test_GT[i])
        print("")
        try:
            img = image_from_url(Data.val_urls[Test_images[Test_mask[i]]])
            plt.imshow(img)
            plt.show()
        except:
            print("error")


def train_step(encoder, decoder, criterion, optimizer):
    Train_totalLoss = 0
    for i in range(int(Train_captions.shape[0] / args.batch_size)):
        start = args.batch_size * i

        Train_caption = Train_captions[start: start + args.batch_size]
        Train_image = Train_images[start: start + args.batch_size]
        Train_length = Train_lengths[start: start + args.batch_size]

        Train_fea = torch.from_numpy(Train_feature[Train_image]).to(device)
        Train_caption = torch.from_numpy(Train_caption).long().to(device)
        Train_length, perm_index = torch.from_numpy(Train_length).sort(0, descending=True)

        cap = Train_caption[perm_index]
        Train_fea = Train_fea[perm_index]

        Train_fea = encoder(Train_fea)

        targets = pack_padded_sequence(cap, Train_length, batch_first=True, enforce_sorted=True)[0]

        outputs = decoder(Train_fea, cap, Train_length)

        loss = criterion(outputs, targets)
        # loss = criterion(outputs.reshape(800, 1004, 17), cap)
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

        Test_fea = torch.from_numpy(Test_feature[Test_image]).to(device)
        Test_caption = torch.from_numpy(Test_caption).long().to(device)
        Test_length, perm_index = torch.from_numpy(Test_length).sort(0, descending=True)

        cap = Test_caption[perm_index]
        Test_fea = Test_fea[perm_index]

        Test_fea = encoder(Test_fea)

        targets = pack_padded_sequence(cap, Test_length, batch_first=True, enforce_sorted=True)[0]

        outputs = decoder(Test_fea, cap, Test_length)

        loss = criterion(outputs, targets)
        # loss = criterion(outputs.reshape(800, 1004, 17), cap)
        Test_totalLoss += loss.item() * args.batch_size

    return Test_totalLoss / Test_captions.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--reg', type=float, default=1e-4)
    args = parser.parse_args()

    print(args)
    main(args)
