import numpy as np
import torch
from collections import defaultdict
from utils import pad_sequences, Dictionary


class Corpus:
    def __init__(self, corpus_dir, w2v, dictionary=None, w2v_lbound=16, w2v_ubound=2 ** 16,
                 corpus_lbound=2, ctx_len=12, pad=0, is_wikitext=True):
        if dictionary is None:
            dictionary = Dictionary(w2v.vector_size)

        if is_wikitext:
            corpus = [fi.lower().split() for fi in (corpus_dir / 'wiki.train.tokens').open().readlines()]
            corpus += [fi.lower().split() for fi in (corpus_dir / 'wiki.valid.tokens').open().readlines()]
            corpus += [fi.lower().split() for fi in (corpus_dir / 'wiki.test.tokens').open().readlines()]
            corpus = np.array(corpus)
        else:
            corpus = []
            for k in [2, 4, 6]:
                with (corpus_dir / f'data.l{k}.txt').open() as f:
                    lines = f.readlines()
                    for l in lines:
                        fields = l.rstrip('\n').split('\t')
                        corpus += [sent.replace('___', ' <unk> ').lower().split() for sent in fields[1].split('@@')]
            corpus = np.unique(corpus)
        print(f"Corpus shape: {corpus.shape}")

        word_count = defaultdict(int)
        for sent in corpus:
            for w in sent:
                word_count[w] += 1
                dictionary.add_word(w, w2v)

        words = []
        for w in dictionary.word2idx:
            if w != '<unk>' and w2v_ubound > w2v.wv.vocab[w].count > w2v_lbound and word_count[w] > corpus_lbound:
                words.append(w)
        print(f"Number of valid words: {len(words)}")

        train_dataset = {}
        valid_dataset = {}
        for w, prob in zip(words, np.random.random(len(words))):
            if prob < 0.9:
                train_dataset[w] = [[], []]
            else:
                valid_dataset[w] = [[], []]

        for sent in corpus:
            words_valid = []
            words_train = []

            for idx, w in enumerate(sent):
                if w in valid_dataset:
                    words_valid += [[w, idx]]
                elif w in train_dataset:
                    words_train += [[w, idx]]

            if len(words_valid) > 0 or len(words_train) > 0:
                sent_word_ids = dictionary.sent2idx(sent)

                if len(words_valid) > 0:
                    for w, idx in words_valid:
                        if np.count_nonzero(sent_word_ids[idx - ctx_len: idx + 1 + ctx_len]) > ctx_len:
                            valid_dataset[w][0] += [sent_word_ids[idx - ctx_len: idx]]
                            valid_dataset[w][1] += [sent_word_ids[idx + 1:  idx + 1 + ctx_len]]

                if len(words_train) > 0:
                    for w, idx in words_train:
                        if np.count_nonzero(sent_word_ids[idx - ctx_len: idx + 1 + ctx_len]) > ctx_len:
                            train_dataset[w][0] += [sent_word_ids[idx - ctx_len: idx]]
                            train_dataset[w][1] += [sent_word_ids[idx + 1:  idx + 1 + ctx_len]]

        for w in valid_dataset:
            lefts = pad_sequences(valid_dataset[w][0], max_len=ctx_len, value=pad, padding='pre', truncating='pre')
            rights = pad_sequences(valid_dataset[w][1], max_len=ctx_len, value=pad, padding='post', truncating='post')
            valid_dataset[w] = np.concatenate((lefts, rights), axis=1)

        for w in train_dataset:
            lefts = pad_sequences(train_dataset[w][0], max_len=ctx_len, value=pad, padding='pre', truncating='pre')
            rights = pad_sequences(train_dataset[w][1], max_len=ctx_len, value=pad, padding='post', truncating='post')
            train_dataset[w] = np.concatenate((lefts, rights), axis=1)

        print(f"Train size: {len(train_dataset.keys())}")
        print(f"Valid size: {len(valid_dataset.keys())}")

        self.dictionary = dictionary
        self.train_dataset = train_dataset
        self.train_words = list(train_dataset.keys())
        self.valid_dataset = valid_dataset
        self.valid_words = list(valid_dataset.keys())
        self.w2v = w2v
        self.ctx_len = ctx_len
        self.train_k2words = {}
        self.valid_k2words = {}

    def _get_words(self, k, use_valid, repeat_ctxs):
        dataset = self.valid_dataset if use_valid else self.train_dataset
        words = self.valid_words if use_valid else self.train_words
        k2words = self.valid_k2words if use_valid else self.train_k2words
        if k not in k2words:
            k2words[k] = [w for w in words if len(dataset[w]) >= k]
        return dataset, words if repeat_ctxs else k2words[k]

    def get_batch(self, batch_size, k_shot, char2idx, device, use_valid=False, fixed=True, repeat_ctxs=False):
        if not fixed:
            k_shot = np.random.randint(k_shot) + 1
        dataset, words = self._get_words(k_shot, use_valid, repeat_ctxs)
        sample_words = np.random.choice(words, batch_size)
        contexts = []
        targets = []
        chars = []
        for word in sample_words:
            sample_sent_idx = np.random.choice(len(dataset[word]), k_shot, replace=repeat_ctxs)
            sample_sents = dataset[word][sample_sent_idx]
            contexts += [sample_sents]
            targets += [self.w2v.wv[word]]
            chars += [[char2idx[c] for c in word if c in char2idx]]
        contexts = torch.tensor(contexts).to(device)
        targets = torch.tensor(targets).to(device)
        chars = torch.tensor(pad_sequences(chars, max_len=2*self.ctx_len)).to(device)
        return contexts, targets, chars
