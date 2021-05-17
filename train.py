# coding=utf-8
from __future__ import print_function

from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.baseline import Transformer
from utils import evaluate, get_lr, load_checkpoint, save_checkpoint, test, train
from config import TrainConfig as C
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT

import os


def build_loaders():
    corpus = None
    if C.corpus == "MSVD":
        corpus = MSVD(C)
    elif C.corpus == "MSR-VTT":
        corpus = MSRVTT(C)
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, C.loader.min_count))
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab


def build_model(vocab):

    model = Transformer(C.feat.size, vocab, C.transformer.d_model, C.transformer.d_ff,
                        C.transformer.n_heads, C.transformer.n_layers, C.transformer.dropout, C.feat.feature_mode)
    if C.pretrained_decoder_fpath is not None:
        model.load_state_dict(torch.load(C.pretrained_decoder_fpath)['transformer'])
        print("Pretrained decoder is loaded from {}".format(C.pretrained_decoder_fpath))

    model.cuda()
    return model


def log_train(summary_writer, e, loss, lr, scores=None):
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    print("loss: {} ".format(loss['total']))

    if scores is not None:
        for metric in C.metrics:
            summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))


def log_val(summary_writer, e, loss, scores):
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
    for metric in C.metrics:
        summary_writer.add_scalar("VAL SCORE/{}".format(metric), scores[metric], e)
    print("loss: {} ".format(loss['total']))
    print("scores: {}".format(scores))


def log_test(summary_writer, e, scores):
    for metric in C.metrics:
        summary_writer.add_scalar("TEST SCORE/{}".format(metric), scores[metric], e)
    print("scores: {}".format(scores))


def main():
    print("MODEL ID: {}".format(C.model_id))

    summary_writer = SummaryWriter(C.log_dpath)

    train_iter, val_iter, test_iter, vocab = build_loaders()
    
    model = build_model(vocab)
    
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    print(get_parameter_number(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay, amsgrad=True)
    # ReduceLROnPlateau自适应调整学习率
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=C.lr_decay_gamma,
                                     patience=C.lr_decay_patience, verbose=True)  # verbose在更新lr后print一个更新信息

    best_val_CIDEr = 0.
    best_epoch = None
    best_ckpt_fpath = None
    for e in range(1, C.epochs + 1):
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)

        """ Train """
        print("\n")
        train_loss = train(e, model, optimizer, train_iter, vocab, C.gradient_clip, C.feat.feature_mode)
        log_train(summary_writer, e, train_loss, get_lr(optimizer))

        """ Validation """
        val_loss = test(model, val_iter, vocab, C.feat.feature_mode)
        val_scores = evaluate(val_iter, model, vocab, C.feat.feature_mode)
        log_val(summary_writer, e, val_loss, val_scores)

        summary_writer.add_scalars("compare_loss/total_loss", {'train_total_loss': train_loss['total'],
                                                               'val_total_loss': val_loss['total']}, e)

        if e >= C.save_from and e % C.save_every == 0:
            print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
            save_checkpoint(e, model, ckpt_fpath, C)

        if e >= C.lr_decay_start_from:
            lr_scheduler.step(val_loss['total'])
        if val_scores['CIDEr'] > best_val_CIDEr:
            best_epoch = e
            best_val_CIDEr = val_scores['CIDEr']
            best_ckpt_fpath = ckpt_fpath

    """ Test with Best Model """
    print("\n\n\n[BEST: {}]".format(best_epoch))
    best_model = load_checkpoint(model, best_ckpt_fpath)
    best_scores = evaluate(test_iter, best_model, vocab, C.feat.feature_mode)
    print("scores: {}".format(best_scores))
    with open("./result/{}.txt".format(C.model_id), 'w') as f:
        f.write(C.model_id + os.linesep)
        f.write("\n\n\n[BEST: {}]".format(best_epoch) + os.linesep)
        f.write("scores: {}".format(best_scores))
        f.write(os.linesep)
    for metric in C.metrics:
        summary_writer.add_scalar("BEST SCORE/{}".format(metric), best_scores[metric], best_epoch)
    save_checkpoint(best_epoch, best_model, C.ckpt_fpath_tpl.format("best"), C)
    summary_writer.close()


if __name__ == "__main__":
    main()

