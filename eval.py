import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


def evaluate(encoder, decoder, features, captions, beam_size, idx_to_word):
    references = list()
    hypotheses = list()
    # features = torch.from_numpy(features)
    # For each image
    for i in range(features.shape[0]):
        k = beam_size
        feature = torch.from_numpy(features[i]).to(device).expand(k, features.shape[1])
        caps = torch.from_numpy(captions[i])

        vocab_size = len(idx_to_word)
        encoder_out = encoder(feature)
        k_prev_words = torch.ones(k).long().to(device)  # (k, 1)

        seqs = k_prev_words  # (k, 1)

        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = decoder.embed(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            inputs = torch.cat([awe, embeddings], dim=1)
            # inputs = embeddings

            h, c = decoder.lstm(inputs, (h, c))  # (s, decoder_dim)

            scores = decoder.linear(h)  # (s, vocab_size)
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
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # print(c.shape)
        # References
        # for w in c :
        #     print(w)

        # print(len(img_caps), c.shape)
        img_captions = []
        # for i in range(caps.shape[0]):
        img_captions = list(
            map(lambda caps: [w for w in c if
                              w not in {idx_to_word['<START>'], idx_to_word['<END>'], idx_to_word['<NULL>']}],
                caps))  # remove <start> and pads
        references.append(img_captions)
        # Hypotheses
        hypotheses.append(
            [w for w in seq if w not in {idx_to_word['<START>'], idx_to_word['<END>'], idx_to_word['<NULL>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4
