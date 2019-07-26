import torch
import torch.nn.functional as F


class Sample:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def caption_image_beam_search(self, encoder_out, idx_to_word, beam_size=3):

        k = beam_size
        vocab_size = len(idx_to_word)

        shape = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, shape)  # (k, encoder_dim)

        k_prev_words = torch.ones(k).long().to(self.device)  # (k, 1)

        seqs = k_prev_words  # (k, 1)

        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

        # Lists to store completed sequences
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h, c = self.decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.decoder.embed(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.decoder.attention(encoder_out, h)  # (s, encoder_dim)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            # inputs = torch.cat([awe, embeddings], dim=1)
            inputs = embeddings

            h, c = self.decoder.lstm(inputs, (awe, c))  # (s, decoder_dim)

            scores = self.decoder.linear(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            if step == 1:
                seqs = seqs[prev_word_inds].reshape(k, 1)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != idx_to_word["<END>"]]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            # seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

        except:
            print("empty")
            seq = 0

        return seq

    def predict(self, image_feature):

        image_feature = torch.from_numpy(image_feature)
        image_feature = image_feature.to(self.device)
        image_feature = self.encoder(image_feature)
        inputs = torch.ones(image_feature.shape[0]).long().to(self.device)

        output = self.decoder.sample(image_feature, inputs)

        return output
