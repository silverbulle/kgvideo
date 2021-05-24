# coding=utf-8
import torch.nn.functional as F
import math
import copy
import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from collections import namedtuple
from gensim.models import KeyedVectors
"""
一般的transformer
"""

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class FeatEmbedding(nn.Module):
    """video feature embedding
        d_feat is per frame dimension
    """
    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)


class TextEmbedding(nn.Module):
    """captioning embedding"""
    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


# class TextEmbedding(nn.Module):
#     """captioning embedding"""
#
#     def __init__(self, vocab, d_model):
#         super(TextEmbedding, self).__init__()
#         self.vocab = vocab
#         self.embed_size = 300
#         self.d_model = d_model
#         weight = self.load_pretrain_weight()
#         self.embed = nn.Embedding.from_pretrained(weight)
#         self.embed.weight.requires_grad = True
#         self.linear = nn.Linear(300, d_model)
#
#     def load_pretrain_weight(self):
#         print "load pretrain word2vec weight[vec dim 300] \n"
#         wvmodel = KeyedVectors.load_word2vec_format("/home/wy/Documents/GoogleNews-vectors-negative300.bin",
#                                                     binary=True)
#         vocab_size = self.vocab.n_vocabs
#         weight = torch.zeros(vocab_size, self.embed_size)
#         for i in range(len(wvmodel.index2word)):
#             try:
#                 index = self.vocab.word2idx[wvmodel.index2word[i]]
#             except:
#                 continue
#             weight[index, :] = torch.from_numpy(wvmodel.get_vector(
#                 self.vocab.idx2word[self.vocab.word2idx[wvmodel.index2word[i]]]))
#         return weight
#
#     def forward(self, x):
#         return self.linear(self.embed(x) * math.sqrt(self.d_model))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dim, dropout, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


def self_attention(query, key, value, dropout=None, mask=None):
    """compute self_attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(scores.size())
    # mask的操作在QK之后，softmax之前
    if mask is not None:
        # mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)  # make mask's index == 1, replace to the value(ex:-1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn


class MultiHeadAttention(nn.Module):
    """compute MultiHead Attention"""
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        # attn是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(Q(K的转置))
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
            mask = mask.unsqueeze(1)
            # print "mask:", mask.size()
        n_batch = query.size(0)
        # linear projections   [10, 32, 512]
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # 变为三维， 或者说是concat head
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)


class PositionWiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output  # + x


class LayerNorm(nn.Module):
    """Construct a layer norm module
        feature = d_model
    """
    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
# eps for guarantee numerical stability,default is 1e-5

class SublayerConnection(nn.Module):
    """sublayer connection : layer norm(x + sublayer(x))
        size = d_model
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        # self.size = size
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))


class EncoderLayer(nn.Module):
    """
    one encoder layer:
    Add & Norm -> Feed Forward -> Add & Norm -> MultiHeadAttention
    """
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # first encoder sublayer
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        # second encoder sublayer
        return self.sublayer_connection[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    one decoder layer
    Sublayer Connection <- Mask MultiHeadAttention(self-attention)
    Sublayer Connection <- MultiHeadAttention(enc-dec attention) <-
    Sublayer Connection <- FeedForward <-
    """

    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x:the input of decoder(captioning)
        :param memory: the output of encoder
        :param src_mask: the input padding mask of encoder
        :param tgt_mask: the input padding mask and sequence mask of decoder
        :return:
        """
        # first block
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, tgt_mask))
        # second block
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    N x EncoderLayer(N=6)
    """
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """
    N x DecoderLayer
    """
    def __init__(self, n, decoder_layer):
        super(Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


def pad_mask(src, trg, pad_idx):
    if isinstance(src, tuple):  # src means features ,if it's tuple, means use different feat
        if len(src) == 3:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            dec_src_mask = dec_src_mask & src_object_mask
            src_mask = (enc_src_mask, dec_src_mask)
            # print('shape of src_maks is ' + str(len(src_mask)))
        if len(src) == 2:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask)
            dec_src_mask = src_image_mask & src_motion_mask  # [32, 1, 50]
            src_mask = (enc_src_mask, dec_src_mask)  # tuple [32, 1, 50]*2
            # print('shape of src_maks is ' + str(len(src_mask)))
    else:
        src_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)
    if trg is not None:  # judge whether it is training mode
        if isinstance(src_mask, tuple):
            trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_image_mask.data)
        else:
            trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)
        return src_mask, trg_mask  # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]

    else:
        return src_mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # np.triu() upper triangle of an array
    return (torch.from_numpy(subsequent_mask) == 0).cuda()


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Transformer(nn.Module):
    """
    the encoder output of the last layer -> each decoder
    """
    def __init__(self, d_feat, vocab, d_model, d_ff, n_heads, n_layers, dropout, feature_mode, device='cuda'):
        super(Transformer, self).__init__()
        self.device = device
        self.vocab = vocab
        self.feature_mode = feature_mode
        if feature_mode == 'one':
            self.src_embed = FeatEmbedding(d_feat, d_model, dropout)
        elif feature_mode == 'two':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        elif feature_mode == 'three':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        self.tgt_embed = TextEmbedding(vocab.n_vocabs, d_model)
        # self.tgt_embed = TextEmbedding(vocab, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(n_layers, EncoderLayer(d_model,
                                                      MultiHeadAttention(n_heads, d_model, dropout),
                                                      PositionWiseFeedForward(d_model, d_ff),
                                                      dropout))
        self.decoder = Decoder(n_layers, DecoderLayer(d_model,
                                                      MultiHeadAttention(n_heads, d_model, dropout),
                                                      PositionWiseFeedForward(d_model, d_ff),
                                                      dropout))
        self.generator = Generator(d_model, vocab.n_vocabs)
        # self.label_smoothing = LabelSmoothing(d_model, vocab.word2idx['<PAD>'], smoothing=0.1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg, mask):
        src_mask, trg_mask = mask
        if self.feature_mode == 'one':
            encoding_outputs = self.encode(src, src_mask)
            output = self.decode(trg, encoding_outputs, src_mask, trg_mask)
        elif self.feature_mode == 'two' or 'three':
            enc_src_mask, dec_src_mask = src_mask
            # print('mask is ' + str(mask) + '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&' + str(type(mask)))  # tuple
            # print('src_mask is ' + str(src_mask) + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' + str(type(src_mask)))  # tuple
            # print('trg_mask is ' + str(trg_mask) + '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@' + str(type(trg_mask)))  # tensor
            # print('enc_src_mask is ' + str(enc_src_mask) + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' + str(type(enc_src_mask)))  # tuple
            # print('dec_src_mask is ' + str(dec_src_mask) + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' + str(type(dec_src_mask)))  # tensor
            encoding_outputs = self.encode(src, enc_src_mask)
            # print('output is ' + str(encoding_outputs) + str(type(encoding_outputs)))  # tensor
            output = self.decode(trg, encoding_outputs, dec_src_mask, trg_mask)
        # elif self.feature_mode == 'three':
        #     enc_src_mask, dec_src_mask = src_mask
        #     encodings_outputs = self.encode(src, enc_src_mask)
        #     output = self.decode(trg, encodings_outputs, dec_src_mask, trg_mask)
        #     print('Wait for thinking!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        pred = self.generator(output)
        # loss = self.label_smoothing(pred.contiguous().view(-1, self.vocab.n_vocabs),
        #                             trg[:, 1:].contiguous().view(-1))
        return pred

    def encode(self, src, src_mask):
        if self.feature_mode == 'one':
            x = self.src_embed(src)
            x = self.pos_embedding(x)
            return self.encoder(x, src_mask)
        elif self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embedding(x1)
            x1 = self.encoder(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embedding(x2)
            x2 = self.encoder(x2, src_mask[1])
            return x1 + x2
        elif self.feature_mode == 'three':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embedding(x1)
            x1 = self.encoder(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embedding(x2)
            x2 = self.encoder(x2, src_mask[1])
            x3 = self.object_src_embed(src[2])
            x3 = self.pos_embedding(x3)
            x3 = self.encoder(x3, src_mask[2])
            # print('----------Encoder x3 len is ' + str(len(x3)))
            return x1 + x2 + x3

    def decode(self, trg, memory, src_mask, trg_mask):
        x = self.tgt_embed(trg)
        x = self.pos_embedding(x)
        return self.decoder(x, memory, src_mask, trg_mask)

    def greed_decode(self, src, max_len):
        batch_size = src.size(0)
        pad_idx = self.vocab.word2idx['<PAD>']
        bos_idx = self.vocab.word2idx['<BOS>']
        with torch.no_grad():
            output = torch.ones(batch_size, 1).fill_(bos_idx).long().cuda()
            src_mask, trg_mask = pad_mask(src[:, :, 0], trg=None, pad_idx=pad_idx)
            memory = self.encode(src, src_mask)
            for i in range(max_len + 2 - 1):
            # for i in range(20 + 1):
                # _, trg_mask = pad_mask(src[:, :, 0], output, pad_idx)
                trg_mask = subsequent_mask(output.size(1))
                dec_out = self.decode(output, memory, src_mask, trg_mask)  # batch, len, d_model
                # pred = self.generator(dec_out)  # batch, len, n_vocabs
                # next_word = pred[:, -1].max(dim=-1)[1].unsqueeze(1)  # pred[:, -1]([batch, n_vocabs])
                pred = self.generator(dec_out[:, -1, :])
                _, next_word = torch.max(pred, dim=1)
                output = torch.cat([output, next_word], dim=-1)
        return output

    def beam_search_decode(self, src, beam_size, max_len):
        """
        An Implementation of Beam Search for the Transformer Model.
        Beam search is performed in a batched manner. Each example in a batch generates `beam_size` hypotheses.
        We return a list (len: batch_size) of list (len: beam_size) of Hypothesis, which contain our output decoded sentences
        and their scores.
        :param src: shape (sent_len, batch_size). Each val is 0 < val < len(vocab_dec). The input tokens to the decoder.
        :param max_len: the maximum length to decode
        :param beam_size: the beam size to use
        :return completed_hypotheses: A List of length batch_size, each containing a List of beam_size Hypothesis objects.
            Hypothesis is a named Tuple, its first entry is "value" and is a List of strings which contains the translated word
            (one string is one word token). The second entry is "score" and it is the log-prob score for this translated sentence.
        Note: Below I note "4 bt", "5 beam_size" as the shapes of objects. 4, 5 are default values. Actual values may differ.
        """
        # 1. Setup
        start_symbol = self.vocab.word2idx['<BOS>']
        end_symbol = self.vocab.word2idx['<EOS>']
    
        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        src_mask = pad_mask(src, trg=None, pad_idx=self.vocab.word2idx['<PAD>'])
        if self.feature_mode == 'one':
            batch_size = src.shape[0]
            model_encodings = self.encode(src, src_mask)
        elif self.feature_mode == 'two':
            batch_size = src[0].shape[0]
            enc_src_mask = src_mask[0]
            dec_src_mask = src_mask[1]
            model_encodings = self.encode(src, enc_src_mask)
        elif self.feature_mode == 'three':
            batch_size = src[0].shape[0]
            enc_src_mask = src_mask[0]
            dec_src_mask = src_mask[1]
            model_encodings = self.encode(src, enc_src_mask)
        
        "model_encodings has shape (batch_size, sentence_len, d_model)"
    
        # 1.2 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        "batch * [beam_size, sentence_len]"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        "batch * []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        ""
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.
    
        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len - 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break
        
            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                if self.feature_mode == 'one':
                    src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
                elif self.feature_mode == 'two':
                    src_mask_l += [dec_src_mask[i:i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'one':
                out = self.decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                  Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device))
            elif self.feature_mode == 'two':
                out = self.decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                  Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device))
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)
            # beam_out = torch.split(out, cur_beam_sizes, dim=0)
            
            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing
            # before other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue
            
                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)
            
                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs,\
                                             top_cand_hyp_pos % self.vocab.n_vocabs
            
                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()
                
                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)
            
                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores
    
        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print('completed_hypotheses', completed_hypotheses)
        return completed_hypotheses
    

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)