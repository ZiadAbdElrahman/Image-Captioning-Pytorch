import torch
import random
from PIL import Image
import argparse
import numpy as np
import torchvision
from torch import nn
from sample import Sample
from DataLoader import DataLoader
from model import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from eval import evaluate
from util import save_weights, print_output, load_weights, image_from_url, idx_to_sentence
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    data = DataLoader(pca=args.PCA, norm=args.norm)

    train_captions, train_feature, train_url, train_len = data.get_Training_data(args.training)
    test_captions, test_feature, test_url, test_len = data.get_val_data(args.testing)
    f, c, _ = data.eval_data()

    writer = SummaryWriter()

    encoder = Encoder(input_size=train_feature.shape[1],
                      hidden_size=args.hidden_size) \
        .to(device)

    decoder = Decoder(embed_size=args.embed_size,
                      hidden_size=args.hidden_size, attention_dim=args.attention_size,
                      vocab_size=len(data.word_to_idx)) \
        .to(device)

    if args.load_weight:
        load_weights(encoder, args.model_path + "Jul28_10-04-57encoder")
        load_weights(decoder, args.model_path + "Jul28_10-04-57decoder")

    for epoch in range(args.num_epochs):
        params = list(decoder.parameters()) + list(encoder.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=params, lr=args.learning_rate)

        # if epoch >= 100:
        training_loss = step(encoder=encoder,
                             decoder=decoder,
                             criterion=criterion,
                             data=(train_captions, train_feature, train_len),
                             optimizer=optimizer)
        # if epoch + 1 % 5 == 0:
        #     a = evaluate(encoder, decoder, train_feature[0:2], train_captions[0:2], 5, data.word_to_idx)
        #     print("bleu4 ", a)

        with torch.no_grad():
            test_loss = step(encoder=encoder,
                             decoder=decoder,
                             criterion=criterion,
                             data=(test_captions, test_feature, test_len))

        # if epoch > 1:
        b1, b2, b3, b4 = evaluate(encoder, decoder, f, c, 5, data.word_to_idx, data.idx_to_word)
        writer.add_scalars('BLEU', {'BLEU1': b1,
                                    'BLEU2': b2,
                                    'BLEU3': b3,
                                    'BLEU4': b4 }, epoch + 1)
        if (epoch % 30) == 0:
            save_weights(encoder, args.model_path + "encoder" + str(epoch))
            save_weights(decoder, args.model_path + "decoder" + str(epoch))

        writer.add_scalars('loss', {'train': training_loss,
                                    'val': test_loss}, epoch + 1)

        print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
              .format(epoch + 1, args.num_epochs, training_loss,
                      np.exp(training_loss), test_loss, np.exp(test_loss)))

        args.learning_rate *= 0.995
        if args.save_weight:
            save_weights(encoder, args.model_path + "encoder"+str(epoch))
            save_weights(decoder, args.model_path + "decoder"+str(epoch))

    if args.save_weight:
        save_weights(encoder, args.model_path + "encoder")
        save_weights(decoder, args.model_path + "decoder")

    if args.predict:

        sample = Sample(encoder=encoder, decoder=decoder, device=device)

        train_mask = [random.randint(0, train_captions.shape[0] - 1) for _ in range(args.numOfpredection)]
        test_mask = [random.randint(0, test_captions.shape[0] - 1) for _ in range(args.numOfpredection)]

        train_featur = torch.from_numpy(train_feature[train_mask])
        train_featur = train_featur.to(device)
        train_encoder_out = encoder(train_featur)

        test_featur = torch.from_numpy(test_feature[test_mask])
        test_featur = test_featur.to(device)
        test_encoder_out = encoder(test_featur)


        train_output = []
        test_output = []

        for i in range(len(test_mask)):
            print(i)
            pre = sample.caption_image_beam_search(train_encoder_out[i].reshape(1, args.embed_size), data.word_to_idx, 2)
            train_output.append(pre)
            pre = sample.caption_image_beam_search(test_encoder_out[i].reshape(1, args.embed_size), data.word_to_idx,
                                                   50)
            test_output.append(pre)

        print_output(output=test_output, sample=0,
                     gt=test_captions[test_mask],
                     img=test_url[test_mask],
                     title="val",
                     show_image=args.show_image,
                     idx_to_word=data.idx_to_word)

        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("")

        print_output(output=train_output, sample=0,
                     gt=train_captions[train_mask],
                     img=train_url[train_mask],
                     title="traning",
                     show_image=args.show_image,
                     idx_to_word=data.idx_to_word)


def step(encoder, decoder, criterion, data, optimizer=None):
    captions, features, lengths = data
    numofstep = int(captions.shape[0] / args.batch_size)
    total_Loss = 0
    for step in range(numofstep):

        start = args.batch_size * step

        caption = torch.from_numpy(captions[start: start + args.batch_size]).long().to(device)
        feature = torch.from_numpy(features[start: start + args.batch_size]).to(device)
        length = torch.from_numpy(lengths[start: start + args.batch_size]).to(device)
        length, sort_ind = length.sort(0, descending=True)

        feature = encoder(feature[sort_ind])
        caption = caption[sort_ind]

        outputs = decoder(feature, caption, length).permute(0, 2, 1)
        # outputs = decoder(feature, caption, length)

        # outputs = pack_padded_sequence(outputs, length, batch_first=True)
        # targets = pack_padded_sequence(caption[:, 1:], length, batch_first=True)

        # loss = criterion(outputs[0], targets[0])
        loss = criterion(outputs, caption[:, 1:])

        total_Loss += (loss.item() * args.batch_size)

        if not optimizer == None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print('step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(step + 1, numofstep, loss.item(),
                              np.exp(loss.item())))
    return total_Loss / (numofstep * args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--load_weight', type=bool, default=False, help='load the weights or not')
    parser.add_argument('--save_weight', type=bool, default=True, help='save the weights or not')
    parser.add_argument('--predict', type=bool, default=False, help='predict random sample or not')
    parser.add_argument('--show_image', type=bool, default=True, help='num of image to predict')
    parser.add_argument('--numOfpredection', type=int, default=50, help='num of image to predict')

    # Data
    parser.add_argument('--PCA', type=bool, default=False, help='PCA the features or not')
    parser.add_argument('--norm', type=bool, default=True, help='normalize the features or not')
    parser.add_argument('--training', type=int, default=None,
                        help='number of images to train on,if None mean all data')
    parser.add_argument('--testing', type=int, default=None, help='number of images to test on, if None mean all data')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--attention_size', type=int, default=512, help='dimension of attention')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--reg', type=float, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
