# coding=utf-8
import os
import time


class MSVDSplitConfig(object):
    # model = "MSVD_InceptionV4"
    model = "MSVD_ResNet152"
    
    video_fpath = "../data/MSVD/features/{}.hdf5".format(model)
    caption_fpath = "../data/MSVD/metadata/MSR Video Description Corpus.csv"

    train_video_fpath = "../data/MSVD/features/{}_train.hdf5".format(model)
    val_video_fpath = "../data/MSVD/features/{}_val.hdf5".format(model)
    test_video_fpath = "../data/MSVD/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "../data/MSVD/metadata/train.csv"
    val_metadata_fpath = "../data/MSVD/metadata/val.csv"
    test_metadata_fpath = "../data/MSVD/metadata/test.csv"


class MSRVTTSplitConfig(object):
    # model = "MSR-VTT_ResNet152"
    model = "MSR-VTT_I3D"

    video_fpath = "../data/MSR-VTT/features/{}.hdf5".format(model)
    train_val_caption_fpath = "../data/MSR-VTT/metadata/train_val_videodatainfo.json"
    test_caption_fpath = "../data/MSR-VTT/metadata/test_videodatainfo.json"

    train_video_fpath = "../data/MSR-VTT/features/{}_train.hdf5".format(model)
    val_video_fpath = "../data/MSR-VTT/features/{}_val.hdf5".format(model)
    test_video_fpath = "../data/MSR-VTT/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "../data/MSR-VTT/metadata/train.json"
    val_metadata_fpath = "../data/MSR-VTT/metadata/val.json"
    test_metadata_fpath = "../data/MSR-VTT/metadata/test.json"
    total_metadata_fpath = "../data/MSR-VTT/metadata/total.json"


class FeatureConfig(object):
    # model = "MSVD_InceptionV4"
    # model = "MSVD_ResNet152"
    # model = "MSR-VTT_ResNet152"
    model = "MSR-VTT_ResNet152+I3D"
    # model = "MSR-VTT_ResNet152+I3D+OFeat"
    # model = "MSVD_ResNet152+I3D"
    # model = "MSVD_InceptionV4+I3D"
    # model = "MSVD_I3D"
    size = None
    feature_mode = None
    # model = models[0]
    if model == 'MSVD_I3D' or model == 'MSR-VTT_I3D':
        size = 1024
        feature_mode = 'one'
    elif model == 'MSVD_ResNet152' or model == 'MSR-VTT_ResNet152':
        size = 2048
        feature_mode = 'one'
    elif model == 'MSVD_InceptionV4' or model == 'MSR-VTT_InceptionV4':
        size = 1536
        feature_mode = 'one'
    elif model == 'MSVD_ResNet152+I3D' or model == 'MSR-VTT_ResNet152+I3D':
        size = [2048, 1024]
        feature_mode = 'two'
    elif model == 'MSVD_InceptionV4+I3D' or model == 'MSR-VTT_InceptionV4+I3D':
        size = [1536, 1024]
        feature_mode = 'two'
    elif model == 'MSR-VTT_ResNet152+I3D+OFeat':
         size = [2048, 1024]
         feature_mode = 'three'
    else:
        raise NotImplementedError("Unknown model: {}".format(model))


class VocabConfig(object):
    # init_word2idx = { '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    init_word2idx = { '<PAD>': 0, '<BOS>': 1, '<EOS>': 2}
    embedding_size = 512


class MSVDLoaderConfig(object):
    n_train = 1200
    n_val = 100
    n_test = 670

    total_caption_fpath = "./data/MSVD/metadata/MSR Video Description Corpus.csv"
    train_caption_fpath = "./data/MSVD/metadata/train.csv"
    val_caption_fpath = "./data/MSVD/metadata/val.csv"
    test_caption_fpath = "./data/MSVD/metadata/test.csv"
    min_count = 1
    max_caption_len = 20

    total_video_feat_fpath_tpl = "./data/{}/features/{}.hdf5"
    phase_video_feat_fpath_tpl = "./data/{}/features/{}_{}.hdf5"
    frame_sampling_method = 'uniform'; assert frame_sampling_method in [ 'uniform', 'random' ]
    # frame_max_len = 100
    frame_sample_len = 50

    num_workers = 6


class MSRVTTLoaderConfig(object):
    n_train = 5175
    n_val = 398
    n_test = 2354

    total_caption_fpath = "./data/MSR-VTT/metadata/total.json"
    train_caption_fpath = "./data/MSR-VTT/metadata/train.json"
    val_caption_fpath = "./data/MSR-VTT/metadata/val.json"
    test_caption_fpath = "./data/MSR-VTT/metadata/test.json"
    min_count = 3
    max_caption_len = 20

    total_video_feat_fpath_tpl = "data/{}/features/{}.hdf5"
    phase_video_feat_fpath_tpl = "data/{}/features/{}_{}.hdf5"
    frame_sampling_method = 'uniform'; assert frame_sampling_method in [ 'uniform', 'random' ]
    frame_max_len = 80
    frame_sample_len = 50

    num_workers = 6


class TransformerConfig(object):
    d_model = 512
    d_ff = 2048
    n_heads = 8
    n_layers = 3
    # n_layers = 4
    dropout = 0.1
    # dropout = 0.3


class EvalConfig(object):
    model_fpath = "/home/wy/PycharmProjects/ABDVC_py2.7/checkpoints/Transformer_baseline | MSVD | FEAT MSVD_ResNet152 mfl-100 fsl-80 mcl-20 | EMB 512 | Transformer d-512-N-2-h-8 | OPTIM AMSGrad lr-0.0001-dc-20-0.2-5-wd-1e-05 rg-0.0 | bs-32 gc-5.0 | Mon Nov 16 20:59:26 2020/best.ckpt"
    result_dpath = "results"
    

class TrainConfig(object):
    corpus = 'MSR-VTT'; assert corpus in [ 'MSVD', 'MSR-VTT' ]
    # corpus = 'MSVD';
    # assert corpus in ['MSVD', 'MSR-VTT']

    feat = FeatureConfig
    vocab = VocabConfig
    loader = {
        'MSVD': MSVDLoaderConfig,
        'MSR-VTT': MSRVTTLoaderConfig
    }[corpus]
    # decoder = DecoderConfig
    transformer = TransformerConfig
    ec = EvalConfig

    """ Optimization """
    epochs = {
        'MSVD': 30,
        'MSR-VTT': 30,
    }[corpus]
    batch_size = 32
    # batch_size = 64
    shuffle = True
    optimizer = "AMSGrad"
    gradient_clip = 5.0  # None if not used
    lr = {
        # 'MSVD': 5e-5,
        'MSVD': 1e-4,
        'MSR-VTT': 2e-4,
    }[corpus]
    lr_decay_start_from = 20
    # lr_decay_gamma = 0.5  # 学习率每次下降0.5倍
    lr_decay_gamma = 0.5  # 学习率每次下降0.5倍
    # lr_decay_patience = 10  # metric停止优化patience个epoch后减小lr
    lr_decay_patience = 10  # metric停止优化patience个epoch后减小lr
    # weight_decay = 1e-5  # L2正则化
    weight_decay = 1e-5

    beam_size = 5
    label_smoothing = 0.1

    """ Pretrained Model """
    pretrained_decoder_fpath = None

    """ Evaluate """
    metrics = [ 'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L' ]

    """ ID """
    exp_id = "Transformer_baseline"
    # feat_id = "FEAT {} mfl-{} fsl-{} mcl-{}".format('+'.join(feat.models), loader.frame_max_len,
    #                                                 loader.frame_sample_len, loader.max_caption_len)
    feat_id = "FEAT {} fsl-{} mcl-{}".format(feat.model, loader.frame_sample_len, loader.max_caption_len)
    
    embedding_id = "EMB {}".format(vocab.embedding_size)
    transformer_id = "Transformer d-{}-N-{}-h-{}".format(transformer.d_model, transformer.n_layers, transformer.n_heads)
    optimizer_id = "OPTIM {} lr-{}-dc-{}-{}-{}-wd-{}".format(
        optimizer, lr, lr_decay_start_from, lr_decay_gamma, lr_decay_patience, weight_decay)
    hyperparams_id = "bs-{}".format(batch_size)
    if gradient_clip is not None:
        hyperparams_id += " gc-{}".format(gradient_clip)

    timestamp = time.asctime(time.localtime(time.time()))
    model_id = " | ".join([ exp_id, corpus, feat_id, embedding_id, transformer_id, optimizer_id, hyperparams_id, timestamp ])

    """ Log """
    log_dpath = "logs_{}/{}".format(feat.model, model_id)
    ckpt_dpath = os.path.join("checkpoints_{}".format(feat.model), model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    save_from = 1
    save_every = 1

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/transformer_CE"
    tx_train_entropy_loss = "loss/train/transformer_reg"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/transformer_CE"
    tx_val_entropy_loss = "loss/val/transformer_reg"
    tx_lr = "params/transformer_LR"




