
from util import load_coco_data
import numpy as np
from util import image_from_url

class DataLoader():
    def __init__(self):
        data = load_coco_data(pca_features=True)

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

    def get_Training_data(self, size):
        # print(self.val_captions.shape, self.val_urls.shape, self.val_image_idxs.shape, self.val_features.shape)
        lengths = np.zeros(size)
        lengths += 17
        for i in range(size):
            for j in range(17):
                if self.train_captions[i, j] == 0:
                    lengths[i] = j
                    break
        img = []
        # for i in range(size):
        #     img.append(self.get_image(self.train_urls[self.train_image_idxs[i]]))
        # img = np.array(img)
        return self.train_captions[0:size], self.train_image_idxs[0:size], self.train_features, lengths, img

    def get_val_data(self, size):
        lengths = np.zeros(size)
        lengths += 17
        for i in range(size):
            for j in range(17):
                if self.val_captions[i, j] == 0:
                    lengths[i] = j
        #             break
        return self.val_captions[0:size], self.val_image_idxs[0:size], self.val_features, lengths

    def next_batch(self, batch_size):
        idx = np.random.choice(self.train_captions.shape[0], batch_size)
        lengths = np.zeros(batch_size)
        lengths += 17
        yield self.train_captions[idx], self.train_image_idxs[idx], self.train_features, lengths

    def get_image(self, img_url):
        img = image_from_url(img_url)
        return np.asarray(img)
