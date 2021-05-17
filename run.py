from __future__ import print_function
import os

import torch

from utils import dict_to_cls, get_predicted_captions, get_groundtruth_captions, save_result, score
from config import EvalConfig as C
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from train import build_model


def run(ckpt_fpath):
    
    checkpoint = torch.load(ckpt_fpath)

    """ Load Config """
    config = dict_to_cls(checkpoint['config'])

    """ Build Data Loader """
    corpus = None
    if config.corpus == "MSVD":
        corpus = MSVD(config)
    elif config.corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    train_iter, val_iter, test_iter, vocab = \
        corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, config.loader.min_count))

    """ Build Models """
    model = build_model(vocab)
    model.load_state_dict(checkpoint['transformer'])

    """ Train Set """
    
    train_vid2pred = get_predicted_captions(train_iter, model, model.vocab)
    train_vid2GTs = get_groundtruth_captions(train_iter, model.vocab)
    train_scores = score(train_vid2pred, train_vid2GTs)
    print("[TRAIN] {}".format(train_scores))
    

    """ Validation Set """
    val_vid2pred = get_predicted_captions(val_iter, model, model.vocab)
    val_vid2GTs = get_groundtruth_captions(val_iter, model.vocab)
    val_scores = score(val_vid2pred, val_vid2GTs)
    print("[VAL] scores: {}".format(val_scores))
    # print(type(train_vid2pred.update(val_vid2pred)), type(train_vid2GTs.update(val_vid2GTs)))
    train_vid2pred.update(val_vid2pred)
    train_vid2GTs.update(val_vid2GTs)
    pesudo_save_fpath = os.path.join(C.result_dpath, "{}_{}.csv".format(config.corpus, 'pesudo_r2l'))
    save_result(train_vid2pred, train_vid2pred, pesudo_save_fpath)

    """ Test Set """
    test_vid2pred = get_predicted_captions(test_iter, model, vocab)
    test_vid2GTs = get_groundtruth_captions(test_iter, vocab)
    test_scores = score(test_vid2pred, test_vid2GTs)
    print("[TEST] {}".format(test_scores))

    test_save_fpath = os.path.join(C.result_dpath, "{}_{}.csv".format(config.corpus, 'test'))
    save_result(test_vid2pred, test_vid2GTs, test_save_fpath)


if __name__ == "__main__":
    run(C.model_fpath)

