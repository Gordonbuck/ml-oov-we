import argparse

parser = argparse.ArgumentParser(description='Training HiCE on WikiText-103')

'''
    Dataset
'''
parser.add_argument('--w2v_dir', type=str, default='../data/base_w2v/wiki_all.sent.split.model',
                    help='location of the default node embedding')
parser.add_argument('--wiki_dir', type=str, default='../data/wikitext-103/',
                    help='location of the training corpus (wikitext-103)')
parser.add_argument('--chimera_dir', type=str, default='../data/chimeras/',
                    help='location of the testing corpus (Chimeras)')
parser.add_argument('--w2v_lbound', type=int, default=16,
                    help='Lower bound of word frequency in w2v for selecting target words')
parser.add_argument('--w2v_ubound', type=int, default=2 ** 16,
                    help='Upper bound of word frequency in w2v for selecting target words')
parser.add_argument('--corpus_lbound', type=int, default=2,
                    help='Lower bound of word frequency in corpus for selecting target words')
parser.add_argument('--ctx_len', type=int, default=12,
                    help='Max length of context either side of target word')
parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')

'''
    Hyperparameters
'''
parser.add_argument('--use_morph', action='store_true',
                    help='use character CNN')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads in self attention')
parser.add_argument('--n_layer', type=int, default=2,
                    help='number of encoding layers')
parser.add_argument('--n_epochs', type=int, default=200,
                    help='upper bound of training epochs')
parser.add_argument('--n_batch', type=int, default=256,
                    help='number of batches in epoch')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr_init', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--n_shot', type=int, default=10,
                    help='upper bound of training K-shot')
parser.add_argument('--fixed_shot', action='store_true',
                    help='fix the K-shot value to n_shot')

'''
    Validation and Test
'''
parser.add_argument('--test_interval', type=int, default=1,
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='../save/',
                    help='location for saving the best model')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='Learning Rate Decay using ReduceLROnPlateau Scheduler')
parser.add_argument('--threshold', type=float, default=1e-3,
                    help='Threshold for ReduceLROnPlateau Scheduler judgement')
parser.add_argument('--patience', type=int, default=4,
                    help='Patience for ReduceLROnPlateau Scheduler judgement')
parser.add_argument('--lr_early_stop', type=float, default=1e-5,
                    help='early stop when lr below this value')

'''
    Meta-Learning
'''
parser.add_argument('--maml', action='store_true',
                    help='adapt to target dataset with 1-st order MAML')
parser.add_argument('--leap', action='store_true',
                    help='adapt to target dataset with leap')
parser.add_argument('--meta_batch_size', type=int, default=128,
                    help='meta batch size')
parser.add_argument('--n_meta_epochs', type=int, default=200,
                    help='upper bound of meta training epochs')
parser.add_argument('--n_meta_batch', type=int, default=16,
                    help='number of batches in meta loop')
parser.add_argument('--n_inner_batch', type=int, default=4,
                    help='number of batches in inner loop for MAML')
parser.add_argument('--n_task_steps', type=int, default=64,
                    help='number of steps to train on a task excluding MAML')
parser.add_argument('--meta_lr_init', type=float, default=5e-4,
                    help='initial learning rate for meta loop')
parser.add_argument('--inner_lr_init', type=float, default=5e-4,
                    help='initial learning rate for inner loop')
