from config import parser
import torch
from hice import HICE
from corpus import Corpus
from gensim.models import Word2Vec
from train import train
from pathlib import Path

if __name__ == '__main__':
    args = args = parser.parse_args()
    print("Loading oracle word embeddings")
    w2v = Word2Vec.load(args.w2v_dir)
    print("Loading Wikitext-103 corpus")
    source_corpus = Corpus(Path(args.wiki_dir), w2v, w2v_lbound=args.w2v_lbound, w2v_ubound=args.w2v_ubound,
                           corpus_lbound=args.corpus_lbound, ctx_len=args.ctx_len)
    char2idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
    device = torch.device(f'cuda:{args.cuda}' if args.cuda != -1 else 'cpu')
    model = HICE(args.n_head, w2v.vector_size, 2 * args.ctx_len, args.n_layer, source_corpus.dictionary.idx2vec,
                 use_morph=args.use_morph)
    print("Training")
    train(model, source_corpus, char2idx, args, device)
