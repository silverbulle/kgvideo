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
import train

import os

train_iter, val_iter, test_iter, vocab = train.build_loaders()
model = train.build_model(vocab)
# del train_iter, val_iter
# Remember to change the mode in config before modifying the load path of ckpt file
ckpt_path = '/home/silverbullet/pyproject/main/checkpoints_MSR-VTT_ResNet152+I3D+OFeat/Transformer_baseline | MSR-VTT | FEAT MSR-VTT_ResNet152+I3D+OFeat fsl-50 mcl-20 | EMB 512 | Transformer d-512-N-3-h-8 | OPTIM AMSGrad lr-0.0002-dc-20-0.5-10-wd-1e-05 | bs-32 gc-5.0 | Wed May 26 15:46:51 2021/17.ckpt'
# ckpt_path ='/home/silverbullet/pyproject/main/checkpoints_MSR-VTT_ResNet152+I3D/Transformer_baseline | MSR-VTT | FEAT MSR-VTT_ResNet152+I3D fsl-50 mcl-20 | EMB 512 | Transformer d-512-N-3-h-8 | OPTIM AMSGrad lr-0.0002-dc-20-0.5-10-wd-1e-05 | bs-32 gc-5.0 | Mon May 17 15:34:33 2021/13.ckpt'

with torch.no_grad():
    best_model = load_checkpoint(model, ckpt_path)
    best_scores = evaluate(test_iter, best_model, vocab, C.feat.feature_mode)
    print("Test scores: {}".format(best_scores))
    with open("./result/{}.txt".format(C.model_id), 'w') as f:
        f.write(C.model_id + os.linesep)
        f.write("\n\n\n[TEST: {}]".format(ckpt_path.split('/')[-1]) + os.linesep)
        f.write("scores: {}".format(best_scores))
        f.write(os.linesep)
    print(best_scores)
    # for metric in C.metrics:
    #     summary_writer.add_scalar("BEST SCORE/{}".format(metric), best_scores[metric], best_epoch)
    # save_checkpoint(best_epoch, best_model, C.ckpt_fpath_tpl.format("best"), C)
    # summary_writer.close()
