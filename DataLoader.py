from util import load_coco_data
import numpy as np


class DataLoader:
    def __init__(self, pca=False, norm=False):
        data = load_coco_data(pca_features=pca)

        self.train_captions = data['train_captions']
        self.train_image_idxs = data['train_image_idxs']
        self.train_features = data['train_features']
        self.train_urls = data['train_urls']
        self.val_captions = data['val_captions']
        self.val_features = data['val_features']
        self.val_image_idxs = data['val_image_idxs']
        self.val_urls = data['val_urls']
        self.idx_to_word = data['idx_to_word']
        self.word_to_idx = data['word_to_idx']

        self.train_size = self.train_captions.shape[0]
        self.val_size = self.val_captions.shape[0]
        if norm:
            mean = np.mean(self.train_features, 0) + 1e-25
            self.train_features -= mean
            self.val_features -= mean
            self.train_features /= mean
            self.val_features /= mean


    def get_Training_data(self, size=None):
        if size == None: size = self.train_size
        lengths = np.zeros(size)
        lengths += 16

        for i in range(size):
            for j in range(16):
                if self.train_captions[i, j] == 0:
                    lengths[i] = j
                    break
        return self.train_captions[:size], self.train_features[self.train_image_idxs[:size]], self.train_urls[
            self.train_image_idxs[:size]], lengths

    def get_val_data(self, size=None):
        if size == None: size = self.val_size
        lengths = np.zeros(size)
        lengths += 17

        for i in range(size):
            for j in range(17):
                if self.train_captions[i, j] == 0:
                    lengths[i] = j
                    break
        return self.val_captions[:size], self.val_features[self.val_image_idxs[:size]], self.val_urls[
            self.val_image_idxs[:size]], lengths
    def eval_data(self):
        ALLcap = np.empty(40000, dtype=np.object)
        ind = np.argsort(self.val_image_idxs)
        # print(self.val_image_idxs[ind])
        # img, sort_ind = self.val_image_idxs.tolist().sort()
        cap = self.val_captions[ind]
        img = self.val_image_idxs[ind]

        for i in range(40000):
            if i % 1000 == 0:
                print(i)
            ALLcap[i] = []
            for j in range(self.val_image_idxs.shape[0]):
                if img[j] == i:
                    sen = []
                    o = self.clean(cap[j])
                    for k in range(len(o)):
                        sen.append(self.idx_to_word[o[k]])
                    ALLcap[i].append(sen)
                if img[j] > i:
                    break
        return self.val_features[0:40000], ALLcap, self.val_urls[0:40000]
    def clean(self, cap):
        clean_cap = []
        for i in range(len(cap)):
            if cap[i] == self.word_to_idx["<START>"] or cap[i] == self.word_to_idx["<NULL>"] or cap[i] == self.word_to_idx["<END>"] :
                continue
            else :
                clean_cap.append(cap[i])
        return clean_cap