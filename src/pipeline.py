from config import parser
import torch
from hice import HICE
from corpus import Corpus
from gensim.models import Word2Vec
from train import train, maml_adapt, leap_adapt
from pathlib import Path
import os
from chimera import Chimeras
from jnlpba import write_word_vecs

if __name__ == '__main__':
    args = args = parser.parse_args()
    print("Loading oracle word embeddings")
    w2v = Word2Vec.load(args.w2v_dir)

    print("Loading Wikitext-103 corpus")
    wiki_corpus = Corpus(Path(args.wiki_dir), w2v, w2v_lbound=args.w2v_lbound, w2v_ubound=args.w2v_ubound,
                         corpus_lbound=args.corpus_lbound, ctx_len=args.ctx_len, is_wikitext=True)
    char2idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
    device = torch.device(f'cuda:{args.cuda}' if args.cuda != -1 else 'cpu')
    model = HICE(args.n_head, w2v.vector_size, 2 * args.ctx_len, args.n_layer, wiki_corpus.dictionary.idx2vec,
                 use_morph=args.use_morph)

    if args.adapt_jnlpba:
        print("Loading JNLPBA corpus")
        target_corpus = Corpus(Path(args.jnlpba_dir), w2v, w2v_lbound=args.w2v_lbound, w2v_ubound=args.w2v_ubound,
                               corpus_lbound=args.corpus_lbound, ctx_len=args.ctx_len,
                               dictionary=wiki_corpus.dictionary, is_jnlpba=True)
    else:
        print("Loading Chimera corpus")
        target_corpus = Corpus(Path(args.chimera_dir), w2v, w2v_lbound=args.w2v_lbound, w2v_ubound=args.w2v_ubound,
                                corpus_lbound=args.corpus_lbound, ctx_len=args.ctx_len,
                                dictionary=wiki_corpus.dictionary, is_chimera=True)
    model.update_embedding(target_corpus.dictionary.idx2vec)

    if args.hice:
        print("Training")
        train(model, wiki_corpus, char2idx, args, device)

    if args.maml:
        print("MAML adaptation")
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt')))
        maml_adapt(model, wiki_corpus, target_corpus, char2idx, args, device)

    if args.leap:
        print("LEAP adaptation")
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt')))
        leap_adapt(model, wiki_corpus, target_corpus, char2idx, args, device)

    if args.adapt_jnlpba:
        print("Exporting word vectors to file")
        for name in ['model', 'maml_model', 'leap_model']:
            model_path = os.path.join(args.save_dir, name + '.pt')
            if not os.path.isfile(model_path):
                continue

            model.load_state_dict(torch.load(model_path))
            model.eval()

            write_word_vecs(model, target_corpus, args.n_shot, char2idx, device, args.oov_wv_dir, name,
                            fixed=args.fixed_shot)
    else:
        print("Loading Chimeras for evaluation")
        chimeras = Chimeras(Path(args.chimera_dir), w2v, wiki_corpus.dictionary, char2idx, ctx_len=args.ctx_len)

        for name in ['model', 'maml_model', 'leap_model']:
            name = os.path.join(args.save_dir, name + '.pt')
            if not os.path.isfile(name):
                continue

            model.load_state_dict(torch.load(name))
            model.eval()

            print(f"{name} evaluation")
            if args.fixed_shot:
                chimeras.eval(model, device, k_shot=args.n_shot)
            else:
                chimeras.eval(model, device)

        if args.fixed_shot:
            chimeras.ground_truth(k_shot=args.n_shot)
        else:
            chimeras.ground_truth()
