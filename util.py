import os
import json
import h5py
import torch
import numpy as np
from builtins import range
from scipy.misc import imread
import matplotlib.pyplot as plt
import urllib.request, urllib.error, urllib.parse, tempfile


def load_weights(model, path):
    model.load_state_dict(torch.load(os.path.join(path)))


def save_weights(model, path):
    torch.save(model.state_dict(), os.path.join(path))


def print_output(output, gt, img, title, show_image, idx_to_word):
    out_sent = idx_to_sentence(output, idx_to_word)
    gt_sent = idx_to_sentence(gt, idx_to_word)
    for i in range(len(out_sent)):
        if show_image:
            try:
                plt.imshow(image_from_url(img[i]))
                plt.title('%s\n%s\nGT:%s' % (title, out_sent[i], gt_sent[i]))
                plt.axis('off')
                plt.show()
            except:
                plt.title('%s\n%s\nGT:%s' % (title, out_sent[i], gt_sent[i]))
                plt.axis('off')
                plt.show()
        else:
            print(out_sent[i])
            print("GT " + gt_sent[i])
            print("")
            print("")


def idx_to_sentence(output, idx_to_word):
    sentences = []
    for i in range(len(output)):
        sentence = ""
        for j in range(16):
            word = idx_to_word[output[i][j]]
            if word == "<END>":
                break
            sentence += " " + word
        sentences.append(sentence)

    return sentences


BASE_DIR = 'coco_captioning'


def load_coco_data(base_dir=BASE_DIR,
                   max_train=None,
                   pca_features=True):
    """
    this fun from cs231n assigment3 files.
    """

    data = {}
    caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data


def image_from_url(url):
    """
    this fun from cs231n assigment3 files.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)
