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
Train_captions, Train_feature, Train_lengths = Data.get_Training_data(150)
Test_captions, Test_feature, Test_lengths = Data.get_val_data(50)

def main(args):
    # writer = SummaryWriter()

    encoder = Encoder(args.embed_size).to(device)
    decoder = Decoder(args.embed_size, args.hidden_size, len(Data.word_to_idx), args.num_layers).to(device)

    # load_weights(encoder, args.model_path + "encoder")
    # load_weights(decoder, args.model_path + "decoder")

    params = list(decoder.parameters()) + list(encoder.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        torch.cuda.empty_cache()

        training_loss = step(encoder, decoder, criterion,
                             (Train_captions, Train_feature, Train_lengths), optimizer)
        torch.cuda.empty_cache()
        with torch.no_grad():
            test_loss = step(encoder, decoder, criterion,
                             (Test_captions, Test_feature, Test_lengths) )

        print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
              .format(epoch + 1, args.num_epochs, training_loss,
                      np.exp(training_loss), test_loss, np.exp(test_loss)))

    save_weights(encoder, args.model_path + "encoder")
    save_weights(decoder, args.model_path + "decoder")

    # load_weights(encoder, args.model_path + "encoder")
    # load_weights(decoder, args.model_path + "decoder")

    sample = Sample(encoder, decoder, device)

    Train_mask = []
    for i in range(10):
        Train_mask.append(random.randint(0, Train_captions.shape[0] - 1))
    Test_mask = []
    for i in range(10):
        Test_mask.append(random.randint(0, Test_captions.shape[0] - 1))

    train_outputS = sample.predict(Train_feature[0:10])
    test_outputS = sample.predict(Test_feature[0:10])

    # train_output = decoder.forward(encoder(torch.from_numpy(Train_feature[Train_images[Train_mask]]).to(device)),
    #                                torch.from_numpy(Train_captions[Train_mask]).long().to(device), 0)
    # test_output = decoder.forward(encoder(torch.from_numpy(Test_feature[Test_images[Test_mask]]).to(device)),
    #                               torch.from_numpy(Test_captions[Test_mask]).long().to(device), 0)

    # _, train_output = train_output.max(2)
    # _, test_output = test_output.max(2)
    # for i in range(len(Train_mask)):
    # print(train_output.shape)
    # _, train_output = train_output.max(2)
    # _, test_output = test_output.max(2)
    # print(pre.shape)
    # outputs.append(pre)
    # outputss = test_output
    # outputss.append(test_output.max(2)[1])
    # outputss = torch.stack(outputss, 1)
    # outputs = torch.stack(outputs, 1)
    # train_sent = idx_to_sentence(train_output, Data.idx_to_word)
    train_sentS = idx_to_sentence(train_outputS, Data.idx_to_word)
    train_GT = idx_to_sentence(Train_captions[0:10], Data.idx_to_word)

    # test_sent = idx_to_sentence(test_output, Data.idx_to_word)
    test_sentS = idx_to_sentence(test_outputS, Data.idx_to_word)
    test_GT = idx_to_sentence(Test_captions[0:10], Data.idx_to_word)

    for i in range(len(Train_mask)):
        # print(train_sent[i])
        print("")
        print(train_sentS[i])
        print(train_GT[i])
        print("")

        # try:
        #     img = image_from_url(Data.train_urls[Train_images[i]])
        #     plt.imshow(img)
        #     plt.show()
        # except:
        #     print("error")

    print("")
    print("")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("")
    print("")

    for i in range(len(Test_mask)):
        # print(test_sent[i])
        print("")
        print(test_sentS[i])
        print(test_GT[i])
        print("")

        # try:
        #     img = image_from_url(Data.val_urls[Test_images[i]])
        #     plt.imshow(img)
        #     plt.show()
        # except:
        #     print("error")


def step(encoder, decoder, criterion, data, optimizer=None):
    captions, features, lengths = data

    total_Loss = 0
    for i in range(int(captions.shape[0] / args.batch_size)):

        start = args.batch_size * i

        caption = torch.from_numpy(captions[start: start + args.batch_size]).long().to(device)
        feature = torch.from_numpy(features[start: start + args.batch_size]).to(device)
        # length, perm_index = torch.from_numpy(lengths[start: start + args.batch_size]).sort(0,
        # descending=True)

        # torch.cuda.empty_cache()

        # caption = caption[perm_index]
        feature = encoder(feature)
        # feature = encoder(feaature[perm_index])

        # torch.cuda.empty_cache()

        # targets = pack_padded_sequence(caption, length, batch_first=True, enforce_sorted=True)[0]

        outputs = decoder(feature, caption, 0)
        # torch.cuda.empty_cache()

        # loss = criterion(outputs, targets)
        loss = criterion(outputs.reshape(25, 1004, 16), caption[:, 1:])
        total_Loss += loss.item() * args.batch_size

        if not optimizer == None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_Loss / captions.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=0)
    args = parser.parse_args()

    print(args)
    main(args)
