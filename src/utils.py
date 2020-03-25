import numpy as np


class Dictionary(object):
    def __init__(self, n_hid):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.idx2vec = [np.zeros(n_hid)]
        self.len = 1

    def add_word(self, word, w2v):
        if word not in self.word2idx and word in w2v.wv:
            self.word2idx[word] = self.len
            self.idx2word += [word]
            self.idx2vec += [w2v.wv[word]]
            self.len += 1

    def __len__(self):
        return self.len

    def idx2sent(self, x):
        return ' '.join([self.idx2word[i] for i in x])

    def sent2idx(self, x):
        return [self.word2idx[w] if w in self.word2idx else 0 for w in x]


def pad_sequences(sequences, max_len=None, padding='pre', truncating='pre', value=0., dtype=np.int32):
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        lengths.append(len(x))

    if max_len is None:
        max_len = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, max_len) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(f'Shape of sample {trunc.shape[1:]} of sequence at position {idx} is different from '
                             f'expected shape {sample_shape}')

        if padding == 'pre':
            x[idx, -len(trunc):] = trunc
        elif padding == 'post':
            x[idx, :len(trunc)] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')

    return x
