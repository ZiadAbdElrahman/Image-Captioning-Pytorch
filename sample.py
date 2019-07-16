import torch

class Sample:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def predict(self, image_feature):

        image_feature = torch.from_numpy(image_feature)
        image_feature = image_feature.to(self.device)
        image_feature = self.encoder(image_feature)

        output = self.decoder.sample(image_feature, None, self.device)

        return output


