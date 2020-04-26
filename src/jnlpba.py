import os
import torch
import random


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


def preprocess_jnlpba(jnlpba_dir, corpus):
    train = 0
    dev = 1
    train_size = 0
    dev_size = 0
    test_size = 0

    print("Writing JNLPBA train and dev files")
    with (jnlpba_dir / "train.txt").open(mode='w+') as f_train:
        with (jnlpba_dir / "dev.txt").open(mode='w+') as f_dev:
            if random.random() < 0.05:
                s = dev
                dev_size += 1
            else:
                s = train
                train_size += 1

            for w in (jnlpba_dir / 'train/Genia4ERtask1.iob2').open().readlines()[:-1]:
                w = w.strip()
                if not (w == '' or len(w.split()) == 2):
                    continue

                if s == train:
                    f_train.write(w + '\n')
                else:
                    f_dev.write(w + '\n')

                if w == '':
                    if random.random() < 0.05:
                        s = dev
                        dev_size += 1
                    else:
                        s = train
                        train_size += 1

    print("Writing JNLPBA test files")
    with (jnlpba_dir / "test.txt").open(mode='w+') as f_test:
        sent = []
        contains_oov = False

        for w in (jnlpba_dir / 'test/Genia4EReval1.iob2').open().readlines():
            w = w.strip()
            if not (w == '' or len(w.split()) == 2):
                continue

            if w == '':
                if contains_oov:
                    test_size += 1
                    f_test.write('\n'.join(sent) + '\n\n')
                sent = []
                contains_oov = False
                continue

            w_0 = w.split()[0].lower()
            if w_0 in corpus.oov_dataset and len(corpus.oov_dataset[w_0] > 0):
                contains_oov = True

            sent += [w]

    print(f'JNLPBA train size: {train_size} dev size: {dev_size} test size: {test_size}')
