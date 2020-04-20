# Write word vectors to file for input to tagger - should be word followed by embedding with word and elements
# separated by spaces. Need to set word_dim for tagger to be that of word embedding
# Split train into train+dev
# Subsample test set to be sentence with at least one OOV word
import os
import torch


def write_word_vecs(model, corpus, k_shot, char2idx, device, oov_wv_dir, model_name, fixed=True):
    model.to(device)
    words, contexts, chars = corpus.get_oov_contexts(k_shot, char2idx, device, fixed=fixed)

    embs = []
    breaks = 2
    offset = 0
    inc = -(-len(words) // breaks)
    with torch.no_grad():
        for b in range(breaks):
            cxts = contexts[offset:offset+inc]
            cs = chars[offset:offset+inc]
            embs += list(model.forward(cxts, cs).cpu().numpy())
            offset += inc

    corpus.w2v.wv.add(words, embs)
    corpus.w2v.wv.save_word2vec_format(os.path.join(oov_wv_dir, f'oov_w2v_{model_name}'))


def preprocess_data():

    return
