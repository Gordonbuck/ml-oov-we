import higher
from leap import Leap
import numpy as np
import os
import torch
import torch.nn as nn


def train(model, source_corpus, char2idx, args, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.patience,
                                                              threshold=args.threshold)
    best_valid_cosine = 1

    for epoch in np.arange(args.n_epochs):
        valid_cosine = []

        model.train()
        for batch in np.arange(args.n_batch):
            train_contexts, train_targets, train_vocabs = source_corpus.get_batch(args.batch_size, args.n_shot,
                                                                                  char2idx, device,
                                                                                  fixed=args.fixed_shot)
            optimizer.zero_grad()
            pred_emb = model.forward(train_contexts, train_vocabs)
            loss = -nn.functional.cosine_similarity(pred_emb, train_targets).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch):
                valid_contexts, valid_targets, valid_vocabs = source_corpus.get_batch(args.batch_size, args.n_shot,
                                                                                      char2idx, device,
                                                                                      use_valid=True,
                                                                                      fixed=args.fixed_shot)
                pred_emb = model.forward(valid_contexts, valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, valid_targets).mean()
                valid_cosine += [loss.cpu().numpy()]

        avg_valid = np.average(valid_cosine)
        lr_scheduler.step(avg_valid)
        print(f"Average valid cosine loss: {avg_valid}")

        if avg_valid < best_valid_cosine:
            best_valid_cosine = avg_valid
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))

        if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('LR early stop')
            break


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
        torch.cuda.max_memory_cached()/ mega_bytes)
    print(string)


def maml_adapt(model, source_corpus, target_corpus, char2idx, args, device):
    model = model.to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, factor=args.lr_decay,
                                                              patience=args.patience, threshold=args.threshold)
    best_score = 3

    for meta_epoch in np.arange(args.n_meta_epochs):
        source_valid_cosine = []
        target_valid_cosine = []

        model.train()
        with torch.backends.cudnn.flags(benchmark=True):
            for meta_batch in np.arange(args.n_meta_batch):
                inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.inner_lr_init)
                meta_optimizer.zero_grad()

                # Have to run inner loop on CPU due to memory leak
                # old_device = device
                # device = torch.device('cpu')
                # model.to(device)

                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                    for inner_batch in np.arange(args.n_inner_batch):
                        source_train_contexts, source_train_targets, source_train_vocabs = source_corpus.get_batch(
                            args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shot)
                        pred_emb = fmodel.forward(source_train_contexts, source_train_vocabs)
                        loss = -nn.functional.cosine_similarity(pred_emb, source_train_targets).mean()
                        diffopt.step(loss)
                        report_memory(f"{meta_epoch},{meta_batch},inner1")

                    target_train_contexts, target_train_targets, target_train_vocabs = target_corpus.get_batch(
                        args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shot)
                    pred_emb = fmodel.forward(target_train_contexts, target_train_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, target_train_targets).mean()
                    loss.backward()
                    report_memory(f"{meta_epoch},{meta_batch},inner2")

                # device = old_device
                # model.to(device)

                meta_optimizer.step()
                report_memory(f"{meta_epoch},{meta_batch},meta")

        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch):
                source_valid_contexts, source_valid_targets, source_valid_vocabs = source_corpus.get_batch(
                    args.meta_batch_size, args.n_shot, char2idx, device, use_valid=True, fixed=args.fixed_shot)
                pred_emb = model.forward(source_valid_contexts, source_valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, source_valid_targets).mean()
                source_valid_cosine += [loss.cpu().numpy()]

                target_valid_contexts, target_valid_targets, target_valid_vocabs = target_corpus.get_batch(
                    args.meta_batch_size, args.n_shot, char2idx, device, use_valid=True, fixed=args.fixed_shot)
                pred_emb = model.forward(target_valid_contexts, target_valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, target_valid_targets).mean()
                target_valid_cosine += [loss.cpu().numpy()]

        avg_source_valid, avg_target_valid = np.average(source_valid_cosine), np.average(target_valid_cosine)
        score = avg_source_valid + avg_target_valid * 2
        lr_scheduler.step(score)
        print(f"Average cosine loss sore: {score}")

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'maml_model.pt'))

        if meta_optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('LR early stop')
            break


def leap_adapt(model, source_corpus, target_corpus, char2idx, args, device):
    model = model.to(device)
    leap = Leap(model)
    meta_optimizer = torch.optim.Adam(leap.parameters(), lr=args.meta_lr_init)
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
            inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.inner_lr_init)
            for inner_batch in np.arange(args.n_task_steps):
                inner_optimizer.zero_grad()
                source_train_contexts, source_train_targets, source_train_vocabs = source_corpus.get_batch(
                        args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shot)
                pred_emb = model.forward(source_train_contexts, source_train_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, source_train_targets).mean()
                loss.backward()
                leap.update(loss, model)
                inner_optimizer.step()

            leap.init_task()
            leap.to(model)
            inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.inner_lr_init)
            for inner_batch in np.arange(args.n_task_steps):
                inner_optimizer.zero_grad()
                target_train_contexts, target_train_targets, target_train_vocabs = target_corpus.get_batch(
                        args.meta_batch_size, args.n_shot, char2idx, device, fixed=args.fixed_shot)
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
                    args.meta_batch_size, args.n_shot, char2idx, device, use_valid=True, fixed=args.fixed_shot)
                pred_emb = model.forward(target_valid_contexts, target_valid_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, target_valid_targets).mean()
                target_valid_cosine += [loss.cpu().numpy()]

        avg_source_valid, avg_target_valid = np.average(source_valid_cosine), np.average(target_valid_cosine)
        score = avg_source_valid + avg_target_valid * 2
        lr_scheduler.step(score)
        print(f"Average cosine loss sore: {score}")

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'leap_model.pt'))

        if meta_optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('LR early stop')
            break
