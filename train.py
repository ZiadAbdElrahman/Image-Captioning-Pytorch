import torch
import random
import argparse
import numpy as np
from torch import nn
from sample import Sample
from DataLoader import DataLoader
from model import Encoder, Decoder
from util import save_weights, print_output, load_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def main(args):
    Data = DataLoader(pca=False, norm=False)
    Train_captions, Train_feature, Train_url = Data.get_Training_data()
    Test_captions, Test_feature, Test_url = Data.get_val_data()

    encoder = Encoder(Train_feature.shape[1], args.hidden_size).to(device)
    decoder = Decoder(args.embed_size, args.hidden_size, len(Data.word_to_idx)).to(device)

    if args.load_weight:
        load_weights(encoder, args.model_path + "encoder")
        load_weights(decoder, args.model_path + "decoder")

    for epoch in range(args.num_epochs):
        params = list(decoder.parameters()) + list(encoder.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=params, lr=args.learning_rate)

        training_loss = step(encoder, decoder, criterion,
                             (Train_captions, Train_feature), optimizer)

        torch.cuda.empty_cache()
        with torch.no_grad():
            test_loss = step(encoder, decoder, criterion,
                             (Test_captions, Test_feature))

        print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
              .format(epoch + 1, args.num_epochs, training_loss,
                      np.exp(training_loss), test_loss, np.exp(test_loss)))

        args.learning_rate *= 0.995

    if args.save_weight:
        save_weights(encoder, args.model_path + "encoder")
        save_weights(decoder, args.model_path + "decoder")

    if args.predict:

        sample = Sample(encoder, decoder, device)
        train_mask = []
        test_mask = []
        for i in range(args.numOfpredection):
            train_mask.append(random.randint(0, Train_captions.shape[0] - 1))
            test_mask.append(random.randint(0, Test_captions.shape[0] - 1))

        train_output = sample.predict(Train_feature[train_mask])
        test_output = sample.predict(Test_feature[test_mask])

        print_output(train_output, Train_captions[train_mask], Train_url[train_mask], show_image=True,
                     idx_to_word=Data.idx_to_word)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print_output(test_output, Test_captions[test_mask], Test_url[test_mask], show_image=True,
                     idx_to_word=Data.idx_to_word)


def step(encoder, decoder, criterion, data, optimizer=None):
    captions, features = data
    numofstep = int(captions.shape[0] / args.batch_size)
    total_Loss = 0
    for i in range(numofstep):

        start = args.batch_size * i

        caption = torch.from_numpy(captions[start: start + args.batch_size]).long().to(device)
        feature = torch.from_numpy(features[start: start + args.batch_size]).to(device)

        feature = encoder(feature)

        outputs = decoder(feature, caption, 0).permute(0, 2, 1)

        loss = criterion(outputs, caption[:, 1:])
        total_Loss += (loss.item() * args.batch_size)

        if i + 1 % 10 == 0:
            print(loss.item())

        if not optimizer == None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_Loss / (numofstep * args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--load_weight', type=bool, default=False, help='load the weights or not')
    parser.add_argument('--save_weight', type=bool, default=True, help='save the weights or not')
    parser.add_argument('--predict', type=bool, default=True, help='predict random sample or not')
    parser.add_argument('--numOfpredection', type=int, default=10, help='nu of image to predict')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--reg', type=float, default=0)
    args = parser.parse_args()

    print(args)
    main(args)
