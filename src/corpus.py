import numpy as np
import torch
from collections import defaultdict
from utils import pad_sequences, Dictionary
import string


class Corpus:
    def __init__(self, corpus_dir, w2v, dictionary=None, w2v_lbound=16, w2v_ubound=2 ** 16,
                 corpus_lbound=2, ctx_len=12, pad=0, is_wikitext=False, is_chimera=False, is_jnlpba=False):
        if dictionary is None:
            dictionary = Dictionary(w2v.vector_size)

        if is_wikitext:
            corpus = [fi.lower().split() for fi in (corpus_dir / 'wiki.train.tokens').open().readlines()]
            corpus += [fi.lower().split() for fi in (corpus_dir / 'wiki.valid.tokens').open().readlines()]
            corpus += [fi.lower().split() for fi in (corpus_dir / 'wiki.test.tokens').open().readlines()]
            corpus = np.array(corpus)
        elif is_chimera:
            corpus = []
            for k in [2, 4, 6]:
                with (corpus_dir / f'data.l{k}.txt').open() as f:
                    lines = f.readlines()
                    for l in lines:
                        fields = l.rstrip('\n').split('\t')
                        corpus += [sent.replace('___', ' <unk> ').lower().split() for sent in fields[1].split('@@')]
            corpus = np.unique(corpus)
        elif is_jnlpba:
            ps = ['train/Genia4ERtask2.iob2', 'test/Genia4EReval2.iob2']
            corpus = []
            sent = []
            for p in ps:
                for w in (corpus_dir / p).open().readlines():
                    if w.startswith("###MEDLINE:"):
                        if sent:
                            corpus += [sent]
                        sent = []
                        continue

                    w = w.strip().lower()
                    if w != '':
                        w = w.split()
                        w = w[0]
                        sent += [w]
                corpus += [sent]
            corpus = np.array(corpus)
        print(f"Corpus shape: {corpus.shape}")

        word_count = defaultdict(int)
        oov_words = []
        oov_dataset = {}
        for sent in corpus:
            for w in sent:
                word_count[w] += 1
                dictionary.add_word(w, w2v)
                if w not in oov_dataset and w not in dictionary.word2idx:
                    if w in string.punctuation:
                        continue
                    oov_words.append(w)
                    oov_dataset[w] = [[], []]

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
            words_oov = []

            for idx, w in enumerate(sent):
                if w in valid_dataset:
                    words_valid += [[w, idx]]
                elif w in train_dataset:
                    words_train += [[w, idx]]
                elif w in oov_dataset:
                    words_oov += [[w, idx]]

            if len(words_valid) > 0 or len(words_train) > 0 or len(words_oov) > 0:
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

                if len(words_oov) > 0:
                    for w, idx in words_oov:
                        if np.count_nonzero(sent_word_ids[idx - ctx_len: idx + 1 + ctx_len]) > ctx_len:
                            oov_dataset[w][0] += [sent_word_ids[idx - ctx_len: idx]]
                            oov_dataset[w][1] += [sent_word_ids[idx + 1:  idx + 1 + ctx_len]]

        for w in valid_dataset:
            lefts = pad_sequences(valid_dataset[w][0], max_len=ctx_len, value=pad, padding='pre', truncating='pre')
            rights = pad_sequences(valid_dataset[w][1], max_len=ctx_len, value=pad, padding='post', truncating='post')
            valid_dataset[w] = np.concatenate((lefts, rights), axis=1)

        for w in train_dataset:
            lefts = pad_sequences(train_dataset[w][0], max_len=ctx_len, value=pad, padding='pre', truncating='pre')
            rights = pad_sequences(train_dataset[w][1], max_len=ctx_len, value=pad, padding='post', truncating='post')
            train_dataset[w] = np.concatenate((lefts, rights), axis=1)

        for w in oov_dataset:
            lefts = pad_sequences(oov_dataset[w][0], max_len=ctx_len, value=pad, padding='pre', truncating='pre')
            rights = pad_sequences(oov_dataset[w][1], max_len=ctx_len, value=pad, padding='post', truncating='post')
            oov_dataset[w] = np.concatenate((lefts, rights), axis=1)

        print(f"Train size: {len(train_dataset.keys())}")
        print(f"Valid size: {len(valid_dataset.keys())}")
        print(f"OOV size: {len(oov_words)}")

        print(f"Train >0 ctxts size: {len([w for w in train_dataset.keys() if len(train_dataset[w]) > 0])}")
        print(f"Valid >0 ctxts size: {len([w for w in valid_dataset.keys() if len(valid_dataset[w]) > 0])}")
        print(f"OOV >0 ctxts size: {len([w for w in oov_words if len(oov_dataset[w]) > 0])}")

        train_ctxt_lens = [len(train_dataset[w]) for w in train_dataset.keys()]
        valid_ctxt_lens = [len(valid_dataset[w]) for w in valid_dataset.keys()]
        oov_ctxt_lens = [len(oov_dataset[w]) for w in oov_words]

        print(f"Number of Train with ctxts size = index {[train_ctxt_lens.count(i) for i in range(10)]}")
        print(f"Number of Valid with ctxts size = index {[valid_ctxt_lens.count(i) for i in range(10)]}")
        print(f"Number of OOV with ctxts size = index {[oov_ctxt_lens.count(i) for i in range(10)]}")

        oov_word_counts = [word_count[w] for w in oov_words]
        inds = np.argsort(-np.array(oov_word_counts))[:10]

        print(f"Number of OOV words with count = index {[oov_word_counts.count(i) for i in range(10)]}")
        print(f"Most frequent OOV words: {[oov_words[i] for i in inds]} "
              f"frequencies: {[oov_word_counts[i] for i in inds]}")

        self.dictionary = dictionary
        self.train_dataset = train_dataset
        self.train_words = list(train_dataset.keys())
        self.valid_dataset = valid_dataset
        self.valid_words = list(valid_dataset.keys())
        self.oov_dataset = oov_dataset
        self.oov_words = list(oov_dataset.keys())
        self.w2v = w2v
        self.ctx_len = ctx_len
        self.train_k2words = {}
        self.valid_k2words = {}

    def _get_words(self, k, use_valid, repeat_ctxs):
        dataset = self.valid_dataset if use_valid else self.train_dataset
        words = self.valid_words if use_valid else self.train_words
        k2words = self.valid_k2words if use_valid else self.train_k2words
        if repeat_ctxs:
            k = 1
        if k not in k2words:
            k2words[k] = [w for w in words if len(dataset[w]) >= k]
        return dataset, k2words[k]

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

    def get_oov_contexts(self, k_shot, char2idx, device, fixed=True):
        if not fixed:
            k_shot = np.random.randint(k_shot) + 1
        dataset = self.oov_dataset
        words = []
        contexts = []
        chars = []
        for word in dataset:
            if fixed and len(dataset[word]) < k_shot:
                continue
            words.append(word)
            sent_idx = np.random.choice(len(dataset[word]), k_shot, replace=not fixed)
            sents = dataset[word][sent_idx]
            contexts += [sents]
            chars += [[char2idx[c] for c in word if c in char2idx]]
        contexts = torch.tensor(contexts).to(device)
        chars = torch.tensor(pad_sequences(chars, max_len=2 * self.ctx_len)).to(device)
        return words, contexts, chars
