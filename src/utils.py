import numpy as np
import torch
from scipy.stats import spearmanr
from pathlib import Path


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


def pad_sequences(sequences, max_len, pre=True, pad=0):
    x = np.full((len(sequences), max_len), pad, dtype=np.int64)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if pre:
            x[idx, -min(max_len, len(s)):] = np.array(s[-max_len:])
        else:
            x[idx, :min(max_len, len(s))] = np.array(s[:max_len])
    return x


def report_memory(name=''):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_cached() / mega_bytes)
    print(string)


def correlate_results(paths, result_path):
    vars = []
    var_to_ind = {}
    inital_lines = paths[0].open().readlines()
    for i in range(len(inital_lines) // 5):
        i = 5 * i
        name_score = inital_lines[i].split()
        name = name_score[0]
        probes = inital_lines[i+1]
        ctx1 = inital_lines[i+2]
        ctx2 = inital_lines[i+3]

        var = [name, probes.split(' '), ctx1, ctx2]
        vars.append(var)
        var_str = ' '.join([name, probes, ctx1, ctx2])
        var_to_ind[var_str] = i // 5

    datas = []
    for p in paths:
        lines = p.open().readlines()
        data = np.zeros(len(lines) // 5)
        for i in range(len(lines) // 5):
            i = 5 * i
            name_score = lines[i].split()
            name = name_score[0]
            score = float(name_score[1])
            probes = lines[i + 1]
            ctx1 = lines[i + 2]
            ctx2 = lines[i + 3]

            var_str = ' '.join([name, probes, ctx1, ctx2])
            data[var_to_ind[var_str]] = score

        datas.append(data)

    cors = []
    for i, d in enumerate(datas):
        for j in range(i+1, len(datas)):
            d2 = datas[j]
            cor = spearmanr(d, d2)[0]
            cors.append(cor)
    print(cors)
    print(np.mean(cors))
    print(np.var(cors))

    data_inds = [np.argsort(data) for data in datas]
    data_ranks = [np.argsort(data) for data in data_inds]
    ranks = np.transpose(data_ranks)
    total_ranks = np.sum(ranks, axis=1)
    inds = np.argsort(total_ranks)

    with result_path.open(mode='w+') as f:
        for i in inds:
            var = vars[i]
            f.write(var[0] + ' ' + str(total_ranks[i]) + ' ' + str(ranks[i]) + '\n')
            f.write(' '.join(var[1]))
            f.write(var[2])
            f.write(var[3])
            f.write('\n')


if __name__ == '__main__':
    paths = ["cr_leap_model_2_2", "cr_leap_model_2_3", "cr_leap_model_2_4", "cr_leap_model_2_5", "cr_leap_model_2_6",
             "cr_maml_model_2_2", "cr_maml_model_2_3", "cr_maml_model_2_4", "cr_maml_model_2_5", "cr_maml_model_2_6",
             "cr_model_2_2", "cr_model_2_3", "cr_model_2_4", "cr_model_2_5", "cr_model_2_6"]
    paths = [Path("../../results/2shot/" + p) for p in paths]
    correlate_results(paths, Path("../../results/2shot_correlated"))
