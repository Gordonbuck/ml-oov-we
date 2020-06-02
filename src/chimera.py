from utils import pad_sequences
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class Chimeras:
    def __init__(self, chimera_dir, w2v, dictionary, char2idx, ctx_len=12, pad=0):
        correct = {}
        with (chimera_dir / 'dataset.txt').open(encoding='latin1') as f:
            for i, line in enumerate(f.readlines()[1:]):
                if i % 2 == 0:
                    nonce = line[:line.find('_')]
                else:
                    correct[nonce] = line.split('\t')[5].split('_')

        fields = ['contexts', 'pivot_vector', 'pivot', 'character', 'probes', 'scores', 'text']
        chimera_data = {}
        for k in [2, 4, 6]:
            chimera_data[k] = {field: [] for field in fields}
            lefts, rights = [], []

            with (chimera_dir / f'data.l{k}.txt').open() as f:
                for l in f.readlines():
                    fields = l.rstrip('\n').split('\t')
                    probe = fields[2].split(',')
                    nonce = fields[0]
                    score = np.array(fields[3].split(','), dtype=np.float)
                    sents = [sent.replace('___', ' ___ ').split() for sent in fields[1].split('@@')]

                    for sent in sents:
                        idx = sent.index('___')
                        lefts += [dictionary.sent2idx(sent[:idx])]
                        rights += [dictionary.sent2idx(sent[idx + 1:])]

                    print(chimera_data[k])
                    chimera_data[k]['pivot_vector'] += [w2v.wv[correct[nonce][0]]]
                    chimera_data[k]['pivot'] += [correct[nonce][0]]
                    chimera_data[k]['character'] += [[char2idx[c] for c in correct[nonce][0] if c in char2idx]]
                    chimera_data[k]['probes'] += [probe]
                    chimera_data[k]['scores'] += [score]
                    chimera_data[k]['text'] += [sents]

            lefts = pad_sequences(lefts, max_len=ctx_len, pad=pad, pre=True)
            rights = pad_sequences(rights, max_len=ctx_len, pad=pad, pre=False)
            chimera_data[k]['contexts'] = np.concatenate((lefts, rights), axis=1).reshape(-1, k, 2*ctx_len)
            chimera_data[k]['character'] = pad_sequences(chimera_data[k]['character'], max_len=ctx_len, pre=True)

        self.chimera_data = chimera_data
        self.w2v = w2v

    def eval(self, model, device, k_shot=None, lang_model=False):
        model.to(device)

        if k_shot is None:
            shots = self.chimera_data
        else:
            shots = [k_shot]

        results = {}
        with torch.no_grad():
            for k_shot in shots:
                results[k_shot] = []
                data = self.chimera_data[k_shot]

                test_contexts = torch.tensor(data['contexts'], dtype=torch.long).to(device)
                test_targets = torch.tensor(data['pivot_vector'], dtype=torch.float).to(device)
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

                inds = np.argsort(spearman_correlations)
                for i in inds:
                    wl = [data['pivot'][i], data['text'][i], spearman_correlations[i], data['probes'][i]]
                    results[k_shot].append(wl)

        return results

    def ground_truth(self, k_shot=None):
        if k_shot is None:
            shots = self.chimera_data
        else:
            shots = [k_shot]

        for k_shot in shots:
            data = self.chimera_data[k_shot]
            pred = data['pivot_vector']

            spearman_correlations = []
            probe_vecs = [[self.w2v.wv[pi] for pi in probe] for probe in data["probes"]]

            for pred, probes, scores in zip(pred, probe_vecs, data["scores"]):
                cos = cosine_similarity([pred], probes)
                cor = spearmanr(cos[0], scores)[0]
                spearman_correlations += [cor]

        print(f"Test with {k_shot} shot: Ground truth embeddings: Spearman: {np.average(spearman_correlations)}")
