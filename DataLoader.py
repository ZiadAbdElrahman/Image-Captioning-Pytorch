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
            mean = np.mean(self.train_features, 0) + 1e-20
            self.train_features -= mean
            self.val_features -= mean
            self.train_features /= mean
            self.val_features /= mean


    def get_Training_data(self, size=None):
        if size == None: size = self.train_size
        return self.train_captions[:size], self.train_features[self.train_image_idxs[:size]], self.train_urls[
            self.train_image_idxs[:size]]

    def get_val_data(self, size=None):
        if size == None: size = self.val_size
        return self.val_captions[:size], self.val_features[self.val_image_idxs[:size]], self.val_urls[
            self.val_image_idxs[:size]]

