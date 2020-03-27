from utils import pad_sequences
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class Chimeras:
    def __init__(self, chimera_dir, w2v, dictionary, char2idx, ctx_len=12, pad=0):
        correct = {}
        with (chimera_dir / 'dataset.txt').open(encoding='latin1') as f:
            ser = 0
            for line in f.readlines()[1:]:
                if ser % 2 == 0:
                    nonce = line[:line.find('_')]
                else:
                    correct[nonce] = line.split('\t')[5].split('_')
                ser += 1

        columns = ['contexts', 'ground_truth_vector', 'target_word', 'character', 'probes', 'scores', 'text']
        chimera_data = {}
        for k in [2, 4, 6]:
            chimera_data[k] = {column: [] for column in columns}
            lefts, rights = [], []

            with (chimera_dir / f'data.l{k}.txt').open() as f:
                lines = f.readlines()

                for l in lines:
                    fields = l.rstrip('\n').split('\t')
                    probe = fields[2].split(',')
                    nonce = fields[0]
                    score = np.array(fields[3].split(','), dtype=np.float)
                    sents = [sent.replace('___', ' ___ ').split() for sent in fields[1].split('@@')]

                    for sent in sents:
                        idx = sent.index('___')
                        lefts += [dictionary.sent2idx(sent[:idx])]
                        rights += [dictionary.sent2idx(sent[idx + 1:])]

                    chimera_data[k]['ground_truth_vector'] += [w2v.wv[correct[nonce][0]]]
                    chimera_data[k]['target_word'] += [correct[nonce][0]]
                    chimera_data[k]['character'] += [[char2idx[c] for c in correct[nonce][0] if c in char2idx]]
                    chimera_data[k]['probes'] += [probe]
                    chimera_data[k]['scores'] += [score]
                    chimera_data[k]['text'] += [sents]

            lefts = pad_sequences(lefts, max_len=ctx_len, value=pad, padding='pre', truncating='pre')
            rights = pad_sequences(rights, max_len=ctx_len, value=pad, padding='post', truncating='post')
            chimera_data[k]['contexts'] = np.concatenate((lefts, rights), axis=1).reshape(-1, k, 2*ctx_len)
            chimera_data[k]['character'] = pad_sequences(chimera_data[k]['character'], max_len=ctx_len)

        self.chimera_data = chimera_data
        self.w2v = w2v

    def eval(self, model, device, k_shot=None):
        if k_shot is None:
            shots = self.chimera_data
        else:
            shots = [k_shot]

        for k_shot in shots:
            data = self.chimera_data[k_shot]

            test_contexts = torch.tensor(data['contexts'], dtype=torch.long).to(device)
            test_targets = torch.tensor(data['ground_truth_vector'], dtype=torch.float).to(device)
            test_vocabs = torch.tensor(data['character'], dtype=torch.long).to(device)

            test_pred = model.forward(test_contexts, test_vocabs)
            cosine = torch.nn.functional.cosine_similarity(test_pred, test_targets).mean().cpu().numpy()

            spearman_correlations = []
            probe_vecs = [[self.w2v.wv[pi] for pi in probe] for probe in data["probes"]]

            for pred, probes, scores in zip(test_pred.cpu().numpy(), probe_vecs, data["scores"]):
                cos = cosine_similarity([pred], probes)
                cor = spearmanr(cos[0], scores)[0]
                spearman_correlations += [cor]

            print(f"Test with {k_shot} shot: Cosine: {cosine};  Spearman: {np.average(spearman_correlations)}")
