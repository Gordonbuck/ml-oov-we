from eval import *
import higher
from leap import Leap
import numpy as np
import os


def train(model, source_corpus, char2idx, args, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.patience,
                                                              threshold=args.threshold)
    best_valid_cosine = -1

    for epoch in np.arange(args.n_epochs):
        valid_cosine = []

        model.train()
        for batch in np.arange(args.n_batch):
            k_shot = np.random.randint(args.n_shot) + 1
            train_contexts, train_targets, train_vocabs = source_corpus.get_batch(args.batch_size, k_shot, char2idx,
                                                                                  device)
            optimizer.zero_grad()
            pred_emb = model.forward(train_contexts, train_vocabs)
            loss = -nn.functional.cosine_similarity(pred_emb, train_targets).mean()
            loss.backward()
            optimizer.step()
            print(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch // args.n_shot):
                for k_shot in np.arange(args.n_shot) + 1:
                    valid_contexts, valid_targets, valid_vocabs = source_corpus.get_batch(args.batch_size, k_shot,
                                                                                          char2idx, device,
                                                                                          use_valid=True)
                    pred_emb = model.forward(valid_contexts, valid_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, valid_targets).mean()
                    valid_cosine += [loss.cpu().numpy()]

        avg_valid = np.average(valid_cosine)
        lr_scheduler.step(avg_valid)

        if avg_valid > best_valid_cosine:
            best_valid_cosine = avg_valid
            torch.save(model, os.path.join(args.save_dir, 'model.pt'))

        if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('LR early stop')
            break


def maml_adapt(model, source_corpus, target_corpus, char2idx, args, device):
    model = model.to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr_init)
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.inner_lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, factor=args.lr_decay,
                                                              patience=args.patience, threshold=args.threshold)
    best_score = -1

    for meta_epoch in np.arange(args.n_epochs):
        source_valid_cosine = []
        target_valid_cosine = []

        model.train()
        for meta_batch in np.arange(args.n_meta_batch):
            meta_optimizer.zero_grad()

            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                for inner_batch in np.arange(args.n_inner_batch):
                    k_shot = np.random.randint(args.n_shot) + 1
                    source_train_contexts, source_train_targets, source_train_vocabs = source_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device)
                    pred_emb = fmodel.forward(source_train_contexts, source_train_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, source_train_targets).mean()
                    diffopt.step(loss)

                k_shot = np.random.randint(args.n_shot) + 1
                target_train_contexts, target_train_targets, target_train_vocabs = target_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device)
                pred_emb = fmodel.forward(target_train_contexts, target_train_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, target_train_targets).mean()
                loss.backward()

            meta_optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch // args.n_shot):
                for k_shot in np.arange(args.n_shot) + 1:
                    source_valid_contexts, source_valid_targets, source_valid_vocabs = source_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device, use_valid=True)
                    pred_emb = model.forward(source_valid_contexts, source_valid_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, source_valid_targets).mean()
                    source_valid_cosine += [loss.cpu().detach().numpy()]

                    target_valid_contexts, target_valid_targets, target_valid_vocabs = target_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device, use_valid=True)
                    pred_emb = model.forward(target_valid_contexts, target_valid_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, target_valid_targets).mean()
                    target_valid_cosine += [loss.cpu().detach().numpy()]

        avg_source_valid, avg_target_valid = np.average(source_valid_cosine), np.average(target_valid_cosine)
        score = avg_source_valid + avg_target_valid * 2
        lr_scheduler.step(score)

        if score > best_score:
            best_score = score
            torch.save(model, os.path.join(args.save_dir, 'maml_model.pt'))

        if meta_optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('LR early stop')
            break


def leap_adapt(model, source_corpus, target_corpus, char2idx, args, device):
    model = model.to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr_init)
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.inner_lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, factor=args.lr_decay,
                                                              patience=args.patience, threshold=args.threshold)
    best_score = -1

    for meta_epoch in np.arange(args.n_epochs):
        source_valid_cosine = []
        target_valid_cosine = []

        model.train()
        leap = Leap(model)
        for meta_batch in np.arange(args.n_meta_batch):
            meta_optimizer.zero_grad()

            leap.init_task()
            leap.to(model)
            for inner_batch in np.arange(args.n_inner_batch):
                inner_optimizer.zero_grad()
                k_shot = np.random.randint(args.n_shot) + 1
                source_train_contexts, source_train_targets, source_train_vocabs = source_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device)
                pred_emb = model.forward(source_train_contexts, source_train_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, source_train_targets).mean()
                loss.backward()
                leap.update(loss, model)
                inner_optimizer.step()

            leap.init_task()
            leap.to(model)
            for inner_batch in np.arange(args.n_inner_batch):
                inner_optimizer.zero_grad()
                k_shot = np.random.randint(args.n_shot) + 1
                target_train_contexts, target_train_targets, target_train_vocabs = target_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device)
                pred_emb = model.forward(target_train_contexts, target_train_vocabs)
                loss = -nn.functional.cosine_similarity(pred_emb, target_train_targets).mean()
                loss.backward()
                leap.update(loss, model)
                inner_optimizer.step()

            leap.normalize()
            meta_optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in np.arange(args.n_batch // args.n_shot):
                for k_shot in np.arange(args.n_shot) + 1:
                    source_valid_contexts, source_valid_targets, source_valid_vocabs = source_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device, use_valid=True)
                    pred_emb = model.forward(source_valid_contexts, source_valid_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, source_valid_targets).mean()
                    source_valid_cosine += [loss.cpu().detach().numpy()]

                    target_valid_contexts, target_valid_targets, target_valid_vocabs = target_corpus.get_batch(
                        args.batch_size, k_shot, char2idx, device, use_valid=True)
                    pred_emb = model.forward(target_valid_contexts, target_valid_vocabs)
                    loss = -nn.functional.cosine_similarity(pred_emb, target_valid_targets).mean()
                    target_valid_cosine += [loss.cpu().detach().numpy()]

        avg_source_valid, avg_target_valid = np.average(source_valid_cosine), np.average(target_valid_cosine)
        score = avg_source_valid + avg_target_valid * 2
        lr_scheduler.step(score)

        if score > best_score:
            best_score = score
            torch.save(model, os.path.join(args.save_dir, 'leap_model.pt'))

        if meta_optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('LR early stop')
            break
