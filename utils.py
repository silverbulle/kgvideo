# coding=utf-8
import inspect
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from models.baseline import pad_mask
from config import TrainConfig as C
from models.label_smoothing import LabelSmoothing

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [[] for _ in range(self.num_losses)]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [0. for _ in range(self.num_losses)]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch, feature_mode):
    if feature_mode == 'one':
        vids, feats, captions = batch
        feats = [feat.cuda() for feat in feats]
        feats = torch.cat(feats, dim=2)
        captions = captions.long().cuda()
        return vids, feats, captions
    elif feature_mode == 'two':
        vids, image_feats, motion_feats, captions = batch
        image_feats = [feat.cuda() for feat in image_feats]
        motion_feats = [feat.cuda() for feat in motion_feats]
        image_feats = torch.cat(image_feats, dim=2)
        motion_feats = torch.cat(motion_feats, dim=2)
        captions = captions.long().cuda()
        feats = (image_feats, motion_feats)
        return vids, feats, captions
    elif feature_mode == 'three':
        vids, image_feats, motion_feats, object_feats, captions = batch
        image_feats = [feat.cuda() for feat in image_feats]
        image_feats = torch.cat(image_feats, dim=2)
        motion_feats = [feat.cuda() for feat in motion_feats]
        motion_feats = torch.cat(motion_feats, dim=2)
        object_feats = [feat.cuda() for feat in object_feats]
        object_feats = torch.cat(object_feats, dim=2)
        captions = captions.long().cuda()
        feats = (image_feats, motion_feats, object_feats)
        return vids, feats, captions
    else:
        raise NotImplementedError("Unknown feature mode: {}".format(feature_mode))


def train(e, model, optimizer, train_iter, vocab, gradient_clip, feature_mode):
    model.train()
    loss_checker = LossChecker(1)
    pad_idx = vocab.word2idx['<PAD>']
    # 定义label smoothing
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    t = tqdm(train_iter)

    for batch in t:
        _, feats, captions = parse_batch(batch, feature_mode)
        trg = captions[:, :-1]
        trg_y = captions[:, 1:]
        norm = (trg_y != pad_idx).data.sum()
        mask = pad_mask(feats, trg, pad_idx)
        optimizer.zero_grad()
        output = model(feats, trg, mask)
        # loss = F.nll_loss(output.view(-1, vocab.n_vocabs),
        #                               trg_y.contiguous().view(-1),
        #                               ignore_index=pad_idx)
        loss = criterion(output.view(-1, vocab.n_vocabs),
                         trg_y.contiguous().view(-1)) / norm
        # entropy_loss = losses.entropy_loss(output, ignore_mask=(trg_y == pad_idx))
        # loss = cross_entropy_loss # + reg_lambda * entropy_loss
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item())  # , cross_entropy_loss.item())#, entropy_loss.item())
        # t.set_description("[Epoch #{0}] loss: {2:.3f} = (CE: {3:.3f}) + (Ent: {1} * {4:.3f})".format(
        #     e, reg_lambda, *loss_checker.mean(last=10)))
        t.set_description("[Epoch #{0}] loss: {1:.3f}".format(e, *loss_checker.mean(last=10)))
        del feats, _, captions
    total_loss = loss_checker.mean()[0]
    loss = {
        'total': total_loss,
    }
    return loss


def test(model, val_iter, vocab, feature_mode):
    model.eval()

    loss_checker = LossChecker(1)
    pad_idx = vocab.word2idx['<PAD>']
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    t = tqdm(val_iter)
    for batch in t:
        _, feats, captions = parse_batch(batch, feature_mode)
        trg = captions[:, :-1]
        trg_y = captions[:, 1:]
        norm = (trg_y != pad_idx).data.sum()
        mask = pad_mask(feats, trg, pad_idx)
        with torch.no_grad():
            output = model(feats, trg, mask)
            loss = criterion(output.view(-1, vocab.n_vocabs),
                             trg_y.contiguous().view(-1)) / norm
            loss_checker.update(loss.item())
        del _, feats, captions
    total_loss = loss_checker.mean()[0]
    loss = {
        'total': total_loss,
    }
    return loss


def get_predicted_captions(data_iter, model, feature_mode):
    def build_onlyonce_iter(data_iter):
        onlyonce_dataset = {}
        for batch in iter(data_iter):
            vids, feats, _ = parse_batch(batch, feature_mode)
            if feature_mode == 'one':
                for vid, feat in zip(vids, feats):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = feat
            elif feature_mode == 'two':
                for vid, image_feat, motion_feat in zip(vids, feats[0], feats[1]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (image_feat, motion_feat)
            elif feature_mode == 'three':
                for vid, image_feat, motion_feat, object_feat in zip(vids, feats[0], feats[1], feats[2]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (image_feat, motion_feat, object_feat)
            # del vids, feats, _
            # print('waiting------------------------, i\'m trying to solve this------')
        onlyonce_iter = []
        vids = list(onlyonce_dataset.keys())
        feats = list(onlyonce_dataset.values())
        # batch_size = 200
        batch_size = 1
        while len(vids) > 0:
            if feature_mode == 'one':
                onlyonce_iter.append((vids[:batch_size], torch.stack(feats[:batch_size])))
            elif feature_mode == 'two':
                image_feats = []
                motion_feats = []
                for image_feature, motion_feature in feats[:batch_size]:
                    image_feats.append(image_feature)
                    motion_feats.append(motion_feature)
                onlyonce_iter.append((vids[:batch_size],
                                      (torch.stack(image_feats), torch.stack(motion_feats))))
            elif feature_mode == 'three':
                image_feats = []
                motion_feats = []
                object_feats = []
                for image_feature, motion_feature, object_feat in feats[:batch_size]:
                    image_feats.append(image_feature)
                    motion_feats.append(motion_feature)
                    object_feats.append(object_feat)
                onlyonce_iter.append((vids[:batch_size],
                                      (torch.stack(image_feats), torch.stack(motion_feats), torch.stack(object_feats))))
            vids = vids[batch_size:]
            feats = feats[batch_size:]
        return onlyonce_iter

    model.eval()

    onlyonce_iter = build_onlyonce_iter(data_iter)

    vid2pred = {}
    # EOS_idx = vocab.word2idx['<EOS>']
    with torch.no_grad():
        for vids, feats in onlyonce_iter:
            # captions = model.greed_decode(feats, C.loader.max_caption_len)
            captions = model.beam_search_decode(feats, C.beam_size, C.loader.max_caption_len)
            captions = [" ".join(caption[0].value) for caption in captions]
            # captions = [for for caption in captions]
            # captions = [ idxs_to_sentence(caption, vocab.idx2word, EOS_idx) for caption in captions ]
            vid2pred.update({v: p for v, p in zip(vids, captions)})
        return vid2pred


def get_groundtruth_captions(data_iter, vocab, feature_mode):
    vid2GTs = {}
    EOS_idx = vocab.word2idx['<EOS>']
    for batch in iter(data_iter):
        vids, _, captions = parse_batch(batch, feature_mode)
        # captions = captions.transpose(0, 1)
        for vid, caption in zip(vids, captions):
            if vid not in vid2GTs:
                vid2GTs[vid] = []
            caption = idxs_to_sentence(caption, vocab.idx2word, EOS_idx)
            vid2GTs[vid].append(caption)
    return vid2GTs


def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = {v: i for i, v in enumerate(vid2pred.keys())}
    refs = {vid2idx[vid]: GTs for vid, GTs in vid2GTs.items()}
    hypos = {vid2idx[vid]: [pred] for vid, pred in vid2pred.items()}

    scores = calc_scores(refs, hypos)
    return scores


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def evaluate(data_iter, model, vocab, feature_mode):
    vid2pred = get_predicted_captions(data_iter, model, feature_mode)
    vid2GTs = get_groundtruth_captions(data_iter, vocab, feature_mode)
    scores = score(vid2pred, vid2GTs)
    return scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


# refers https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.load_state_dict(checkpoint['transformer'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'transformer': model.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            if len(vid2GTs[vid]) == 1:
                GTs = vid2GTs[vid]
            else:
                GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([str(vid), pred, GTs])
            fout.write("{}\n".format(line))
