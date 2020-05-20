import higher
from leap import Leap
import numpy as np
import os
import torch
import torch.nn as nn
import gc


def train(model, source_corpus, char2idx, args, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.patience,
                                                              threshold=args.threshold)
    best_valid_cosine = 1

    for epoch in np.arange(args.n_epochs):
        valid_cosine = []
        valid_ce = []

        model.train()
        for batch in np.arange(args.n_batch):
            train_contexts, train_targets, train_vocabs, train_inds = source_corpus.get_batch(args.batch_size,
                                                                                              args.n_shot,
                                                                                              char2idx, device,
                                                                                              fixed=args.fixed_shot,
                                                                                              return_inds=True)
            optimizer.zero_grad()

            if args.active_learning:
                pred_emb, pred_ind = model.forward(train_contexts, train_vocabs, train_lang_model=args.active_learning)
                loss = nn.functional.cross_entropy(pred_ind, train_inds)
                loss.backward()
            else:
                pred_emb = model.forward(train_contexts, train_vocabs)

            loss = -nn.functional.cosine_similarity(pred_emb, train_targets).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch):
                valid_contexts, valid_targets, valid_vocabs, valid_inds = source_corpus.get_batch(args.batch_size,
                                                                                                  args.n_shot,
                                                                                                  char2idx, device,
                                                                                                  use_valid=True,
                                                                                                  fixed=args.fixed_shot,
                                                                                                  return_inds=True)
                if args.active_learning:
                    pred_emb, pred_ind = model.forward(valid_contexts, valid_vocabs,
                                                       train_lang_model=args.active_learning)
                    loss = nn.functional.cross_entropy(pred_ind, valid_inds).mean()
                    valid_ce += [loss.cpu().numpy()]
                else:
                    pred_emb = model.forward(valid_contexts, valid_vocabs)

                loss = -nn.functional.cosine_similarity(pred_emb, valid_targets).mean()
                valid_cosine += [loss.cpu().numpy()]

        avg_valid = np.average(valid_cosine)
        lr_scheduler.step(avg_valid)

        if args.active_learning:
            avg_ce = np.average(valid_ce)
            print(f"Average cosine loss: {avg_valid}; Average cross entropy loss: {avg_ce}")
        else:
            print(f"Average cosine loss: {avg_valid}")

        if avg_valid < best_valid_cosine:
            best_valid_cosine = avg_valid
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))

        if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('LR early stop')
            break


def maml_adapt(model, source_corpus, target_corpus, char2idx, args, device):
    model = model.to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.maml_meta_lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, factor=args.lr_decay,
                                                              patience=args.patience, threshold=args.threshold)
    best_score = 3

    for meta_epoch in np.arange(args.n_meta_epochs):
        gc.collect()
        source_valid_cosine = []
        target_valid_cosine = []

        model.train()
        with torch.backends.cudnn.flags(benchmark=True):
            for meta_batch in np.arange(args.n_meta_batch):
                inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.maml_inner_lr_init)
                meta_optimizer.zero_grad()

                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                    for inner_batch in np.arange(args.n_inner_batch):
                        source_train_contexts, source_train_targets, source_train_vocabs = source_corpus.get_batch(
                            args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shot)
                        pred_emb = fmodel.forward(source_train_contexts, source_train_vocabs)
                        loss = -nn.functional.cosine_similarity(pred_emb, source_train_targets).mean()
                        diffopt.step(loss)

                    target_train_contexts, target_train_targets, target_train_vocabs = target_corpus.get_batch(
                        args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shot,
                        repeat_ctxs=args.meta_repeat_ctxs, lang_model=model if args.active_learning else None)
                    pred_emb = fmodel.forward(target_train_contexts, target_train_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, target_train_targets).mean()
                    loss.backward()

                meta_optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch):
                source_valid_contexts, source_valid_targets, source_valid_vocabs = source_corpus.get_batch(
                    args.meta_batch_size, args.n_shot, char2idx, device, use_valid=True, fixed=args.fixed_shot)
                pred_emb = model.forward(source_valid_contexts, source_valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, source_valid_targets).mean()
                source_valid_cosine += [loss.cpu().numpy()]

                target_valid_contexts, target_valid_targets, target_valid_vocabs = target_corpus.get_batch(
                    args.meta_batch_size, args.n_shot, char2idx, device, use_valid=True, fixed=args.fixed_shot,
                    repeat_ctxs=args.meta_repeat_ctxs)
                pred_emb = model.forward(target_valid_contexts, target_valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, target_valid_targets).mean()
                target_valid_cosine += [loss.cpu().numpy()]

        avg_source_valid, avg_target_valid = np.average(source_valid_cosine), np.average(target_valid_cosine)
        score = avg_target_valid
        lr_scheduler.step(score)
        print(f"Average source cosine loss: {avg_source_valid}; Average target cosine loss: {avg_target_valid}")

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'maml_model.pt'))

        if meta_optimizer.param_groups[0]['lr'] < args.maml_lr_early_stop:
            print('LR early stop')
            break


def leap_adapt(model, source_corpus, target_corpus, char2idx, args, device):
    model = model.to(device)
    leap = Leap(model)
    meta_optimizer = torch.optim.Adam(leap.parameters(), lr=args.leap_meta_lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, factor=args.lr_decay,
                                                              patience=args.patience, threshold=args.threshold)
    best_score = 3

    for meta_epoch in np.arange(args.n_meta_epochs):
        source_valid_cosine = []
        target_valid_cosine = []

        model.train()
        for meta_batch in np.arange(args.n_meta_batch):
            meta_optimizer.zero_grad()

            leap.init_task()
            leap.to(model)
            inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.leap_inner_lr_init)
            for inner_batch in np.arange(args.n_task_steps):
                inner_optimizer.zero_grad()
                source_train_contexts, source_train_targets, source_train_vocabs = source_corpus.get_batch(
                        args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shots)
                pred_emb = model.forward(source_train_contexts, source_train_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, source_train_targets).mean()
                loss.backward()
                leap.update(loss, model)
                inner_optimizer.step()

            leap.init_task()
            leap.to(model)
            inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.leap_inner_lr_init)
            for inner_batch in np.arange(args.n_task_steps):
                inner_optimizer.zero_grad()
                target_train_contexts, target_train_targets, target_train_vocabs = target_corpus.get_batch(
                    args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shot,
                    repeat_ctxs=args.meta_repeat_ctxs, lang_model=model if args.active_learning else None)
                pred_emb = model.forward(target_train_contexts, target_train_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, target_train_targets).mean()
                loss.backward()
                leap.update(loss, model)
                inner_optimizer.step()

            leap.normalize()
            meta_optimizer.step()

        leap.to(model)
        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch):
                source_valid_contexts, source_valid_targets, source_valid_vocabs = source_corpus.get_batch(
                    args.meta_batch_size, args.n_shot, char2idx, device, use_valid=True, fixed=args.fixed_shot)
                pred_emb = model.forward(source_valid_contexts, source_valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, source_valid_targets).mean()
                source_valid_cosine += [loss.cpu().numpy()]

                target_valid_contexts, target_valid_targets, target_valid_vocabs = target_corpus.get_batch(
                    args.meta_batch_size, args.n_shot, char2idx, device, use_valid=True, fixed=args.fixed_shot,
                    repeat_ctxs=args.meta_repeat_ctxs)
                pred_emb = model.forward(target_valid_contexts, target_valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, target_valid_targets).mean()
                target_valid_cosine += [loss.cpu().numpy()]

        avg_source_valid, avg_target_valid = np.average(source_valid_cosine), np.average(target_valid_cosine)
        score = avg_target_valid
        lr_scheduler.step(score)
        print(f"Average source cosine loss: {avg_source_valid}; Average target cosine loss: {avg_target_valid}")

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'leap_model.pt'))

        if meta_optimizer.param_groups[0]['lr'] < args.leap_lr_early_stop:
            print('LR early stop')
            break
