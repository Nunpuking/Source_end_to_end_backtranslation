
import argparse
import numpy as np
import random
import time
import os
import math
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import torch.distributed as dist
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.autograd import Variable
import copy

PAD = 0
BOS = 1
EOS = 2
UNK = 3
CTC = 4

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
CTC_WORD = '<ctcblank>'

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def CudaVariable(X):
    return Variable(X).to(device)

class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, epsilon):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, target):
        smoothing = self.epsilon
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss

def get_scale(nin, nout):
    return math.sqrt(6)/math.sqrt(nin+nout) # Xavier

def one_hot(input, class_tensor, num_classes):
    Bn, Tx = input.size()
    input = input.reshape(-1).unsqueeze(1)
    if input.dtype == torch.long:
        data_type = torch.float
    else:
        data_type = input.dtype
    return (input == class_tensor.reshape(1, num_classes)).type(data_type).view(Bn, Tx, -1)

class myEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(myEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.class_tensor = torch.arange(num_embeddings).cuda()

    def forward(self, input):
        # input : Bn, Tx
        if len(input.size()) == 2:
            Bn, Tx = input.size()
            input = one_hot(input, self.class_tensor, self.num_embeddings) # Bn*Tx, C
        else:
            Bn, Tx, _ = input.size()
        input = input.reshape(Bn*Tx, -1)
        out = torch.mm(input, self.weight) # Bn*Tx, Emb
        return out.view(Bn, Tx, -1)

    def reset_parameters(self):
        scale = get_scale(1, self.embedding_dim)
        self.weight.data.uniform_(-scale, scale)

class myLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(myLinear, self).__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        if self.in_features == self.out_features: # Identity
            self.weight.data.copy_(torch.eye(self.in_features))
        else:
            scale = get_scale(self.in_features, self.out_features)
            self.weight.data.uniform_(-scale, scale)

        if self.bias is not None:
            self.bias.data.zero_()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk, drop_p=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = float(dk)# ** 0.5 # this is better without 0.5
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, q, k, v, mask=None): # B H T E
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temper # B H Tq Tk
        if mask is not None:
            attn = attn.masked_fill(mask<0.1, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) # B H Tq Tk 
        output = torch.matmul(attn, v) # B H Tv E 
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim_model, dk, dv, drop_p=0., resnorm_type='norm_res'):
        super(MultiHeadAttention, self).__init__()

        self.n_head, self.dk, self.dv = n_head, dk, dv

        self.mat_qs = myLinear(dim_model, n_head*dk, bias=False)
        self.mat_ks = myLinear(dim_model, n_head*dk, bias=False)
        self.mat_vs = myLinear(dim_model, n_head*dv, bias=False)

        self.sdp_attn = ScaledDotProductAttention(dk, drop_p=drop_p)

        self.out_proj = myLinear(n_head*dv, dim_model)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(dim_model)

        self.resnorm_type = resnorm_type

    def forward(self, q, k, v, attn_mask=None): # (QK')V # q = B T E
        n_head, dk, dv = self.n_head, self.dk, self.dv
        Bn, Tq, Tk, Tv = q.size(0), q.size(1), k.size(1), v.size(1)

        # TODO: merge the matrices and project q,k,v at the same time and split
        qnew = self.mat_qs(q).view(Bn, Tq, n_head, dk).transpose(1,2)
        k = self.mat_ks(k).view(Bn, Tk, n_head, dk).transpose(1,2)
        v = self.mat_vs(v).view(Bn, Tv, n_head, dv).transpose(1,2)

        if attn_mask is not None: #  B ? T -> B ? ? T 
            attn_mask = attn_mask.unsqueeze(1)

        output, attn = self.sdp_attn(qnew, k, v, mask=attn_mask) # Bn H Ty E
        output = output.transpose(1, 2).contiguous().view(Bn, Tq, -1) # Bn Ty H*E
        output = self.dropout(self.out_proj(output))

        if self.resnorm_type == 'norm_res':
            return (self.layer_norm(output) + q), attn
        elif self.resnorm_type == 'res_norm':
            return self.layer_norm(output+q), attn

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, drop_p=0., resnorm_type='norm_res'):
        super().__init__()
        self.layer1 = myLinear(dim_model, dim_ff)
        self.layer2 = myLinear(dim_ff, dim_model)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.resnorm_type = resnorm_type

    def forward(self, x):
        output = self.layer2(F.relu(self.layer1(x)))
        output = self.dropout(output)
        if self.resnorm_type == 'norm_res':
            return self.layer_norm(output) + x
        elif self.resnorm_type == 'res_norm':
            return self.layer_norm(output + x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len + 2 #BOS, EOS tokens
        self.get_pe(self.max_len)

    def forward(self, x, coeff=1.0):
        if self.training or x.size(1) <= self.max_len:
            pass
        else:
            self.get_pe(x.size(1))
        x = x + coeff*self.pe[:, :x.size(1)]
        return self.dropout(x)

    def get_pe(self, max_len):
        scale = get_scale(1, self.d_model)
        pe = torch.zeros(max_len, self.d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position*div_term) * scale
        pe[:, 1::2] = torch.cos(position*div_term) * scale
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

class TM_EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0., resnorm_type='norm_res'):
        super(TM_EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p,\
                                             resnorm_type=resnorm_type)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p, resnorm_type=resnorm_type)

    def forward(self, enc_in, x_mask=None):
        enc_out, attn = self.self_attn(enc_in, enc_in, enc_in, attn_mask=x_mask)
        enc_out = self.ff_layer(enc_out)
        return enc_out, attn

class TM_Encoder(nn.Module):
    def __init__(self, src_words_n, n_layers=6, n_head=8, dk=64, dv=64,
                    dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0., max_len=250,\
                    resnorm_type='norm_res'):
        super(TM_Encoder, self).__init__()

        # first in : 
        self.src_emb = myEmbedding(src_words_n, dim_wemb)#, padding_idx=PAD)
        self.pos_enc = PositionalEncoding(dim_wemb, max_len, drop_p)
        self.layer_norm = nn.LayerNorm(dim_wemb)
        # repeated layer
        self.layer_stack = nn.ModuleList([
            TM_EncoderLayer(dim_model, dim_ff, n_head, dk, dv, drop_p=drop_p,\
                             resnorm_type=resnorm_type) for _ in range(n_layers)])

    def forward(self, src_seq, src_mask):
        src_mask = src_mask.unsqueeze(1) # B 1 T
        enc_out = self.src_emb(src_seq) # Word embedding look up # Bn Tx Emb
        enc_out = self.pos_enc(enc_out) # Position Encoding
        enc_out = self.layer_norm(enc_out)

        for enc_layer in self.layer_stack:
            enc_out, _ = enc_layer(enc_out, x_mask=src_mask)
        return enc_out

class TM_DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0., resnorm_type='norm_res'):
        super(TM_DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p,\
                                            resnorm_type=resnorm_type)
        self.enc_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p,\
                                            resnorm_type=resnorm_type)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p, resnorm_type=resnorm_type)

    def forward(self, dec_in, enc_out, y_mask=None, enc_mask=None):
        dec_out, self_attn = self.self_attn(dec_in, dec_in, dec_in, attn_mask=y_mask)
        dec_out, cross_attn = self.enc_attn(dec_out, enc_out, enc_out, attn_mask=enc_mask)
        dec_out = self.ff_layer(dec_out)
        return dec_out, self_attn, cross_attn

class TM_Decoder(nn.Module):
    def __init__(self, trg_words_n, n_layers=6, n_head=8, dk=64, dv=64,
            dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0., max_len=250,\
            resnorm_type='norm_res'):
        super(TM_Decoder, self).__init__()

        # first in
        self.dec_emb = myEmbedding(trg_words_n, dim_wemb)#, padding_idx=PAD)
        self.pos_enc = PositionalEncoding(dim_wemb, max_len, drop_p)
        self.layer_norm = nn.LayerNorm(dim_wemb)
        # repeated layer
        self.layer_stack = nn.ModuleList([
            TM_DecoderLayer(dim_model, dim_ff, n_head, dk, dv, drop_p=drop_p,\
                             resnorm_type=resnorm_type) for _ in range(n_layers)])
        #self.trg_word_proj.weight = self.dec_emb.weight # Share the weight 

    def get_subsequent_mask(self, seq):
        if len(seq.size()) == 2: # seq is scalar values
            s0, s1 = seq.size()
        else:
            s0, s1, _ = seq.size() # seq is one-hot processed values
        mask = torch.tril(torch.ones((s1, s1), device=seq.device), diagonal=0)
        return mask.type(torch.cuda.FloatTensor).unsqueeze(0)

    def forward(self, trg_seq, trg_mask, enc_out, src_mask):
        src_mask = src_mask.unsqueeze(1) # B 1 T
        trg_mask = trg_mask.unsqueeze(1) # B 1 T
        dec_out = self.dec_emb(trg_seq) # Word embedding look up
        dec_out = self.pos_enc(dec_out) # Posision Encoding
        dec_out = self.layer_norm(dec_out)


        mh_attn_sub_mask = self.get_subsequent_mask(trg_seq) # lower traingle matrix 
        y_mask = trg_mask * mh_attn_sub_mask

        for dec_layer in self.layer_stack:
            dec_out, _, _ = dec_layer(dec_out, enc_out, y_mask=y_mask, enc_mask=src_mask)

        return dec_out # B Ty E

class BT_Transformer(nn.Module):
    def __init__(self, args=None):
        super(BT_Transformer, self).__init__()
        self.gen_method = args.gen_method
        self.sampling_ratio = args.sampling_ratio
        self.k = args.n_topk_sampling
        self.class_tensor = torch.arange(args.trg_words_n).cuda()
        self.trg_words_n = args.trg_words_n
        self.max_len = 150 # 121 is fairseq's maximum sentence length considers during test in IWSLT2014 (there are longer sentence but filtered out)

        resnorm_type = getattr(args, 'tm_resnorm_type', 'norm_res')
        src_words_n, trg_words_n = args.src_words_n, args.trg_words_n
        dim_wemb, dim_model = args.dim_wemb, args.dim_model
        drop_p = args.dropout_p
        dim_ff, n_layers = args.tm_dim_ff, args.tm_n_layers
        n_head, dk, dv = args.tm_n_head, args.tm_dk, args.tm_dv
        assert dim_model == dim_wemb, 'dim_model == dim_wemb for residual connections'

        self.encoder=TM_Encoder(src_words_n, n_layers=n_layers, n_head=n_head, dk=dk, dv=dv,
                dim_wemb=dim_wemb, dim_model=dim_model, dim_ff=dim_ff, drop_p=drop_p,
                emb_noise=args.emb_noise, max_len=self.max_len, resnorm_type=resnorm_type)
        self.decoder=TM_Decoder(trg_words_n, n_layers=n_layers, n_head=n_head, dk=dk, dv=dv,
                dim_wemb=dim_wemb, dim_model=dim_model, dim_ff=dim_ff, drop_p=drop_p,
                emb_noise=args.emb_noise, max_len=self.max_len, resnorm_type=resnorm_type)
        self.logit_layer = nn.Linear(dim_model, trg_words_n)


        if args.joined_dictionary == 1:
            self.decoder.dec_emb.weight = self.encoder.src_emb.weight

        if args.label_smoothing > 0.0:
            self.criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.nll = nn.NLLLoss(reduction='none')
        self.reparam_method = getattr(args, 'bt_reparam_method', 'crt')


    def forward(self, x_data, x_mask, y_data, y_mask):

        y_target = y_data[:,1:] # label
        if y_target.dtype != torch.long:
            y_target = y_target.type(torch.long)
        y_mask = y_mask[:,1:]
        y_in = y_data[:,:-1] # input as teacher forcing
        Bn, Ty = y_in.size()

        # encode and decode
        enc_out = self.encoder(x_data, x_mask) # B Tx E
        dec_out = self.decoder(y_in, y_mask, enc_out, x_mask) # B Ty E(num of words)
        out = self.logit_layer(dec_out)

        # loss
        loss = self.criterion(out.view(-1, out.size(2)), y_target.contiguous().view(-1))
        loss = loss * y_mask.contiguous().view(-1)

        return loss

    def decode_forward(self, enc_out, x_mask, y_data, y_mask):
        # Instead of x_data, it receives processed enc_out

        y_target = y_data[:,1:] # label
        if y_target.dtype != torch.long:
            y_target = y_target.type(torch.long)
        y_mask = y_mask[:,1:]
        y_in = y_data[:,:-1] # input as teacher forcing
        Bn, Ty = y_in.size()

        # encode and decode
        dec_out = self.decoder(y_in, y_mask, enc_out, x_mask) # B Ty E(num of words)
        out = self.logit_layer(dec_out)

        # loss
        loss = self.criterion(out.view(-1, out.size(2)), y_target.contiguous().view(-1))
        loss = loss * y_mask.contiguous().view(-1)

        return loss


    def org_gumbel_softmax(self, logits, tau=0.1, dim=-1, hard=False, grad_coeff=1.0):
        gumbels = (-torch.empty_like(logits, memory_format=\
                                            torch.legacy_contiguous_format).exponential_().log() )
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)

        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=\
                                            torch.legacy_contiguous_format).scatter_(dim,index,1.0)
            out = y_hard - grad_coeff*y_soft.detach() + grad_coeff*y_soft
        else:
            out = y_soft
        return out

    def index_gumbel_softmax(self, logits, index, tau=0.1, dim=-1, hard=False, grad_coeff=1.0):
        # logits : Bn*T, C
        # index  : Bn*T, 1
        _, C = logits.size()

        min_logits = torch.min(logits, dim=-1).values
        max_logits = torch.max(logits, dim=-1).values
        diff_logits = max_logits-min_logits + tau

        one_hot_indices = one_hot(index, self.class_tensor, C).reshape(-1, C)
        indexed_logits = (logits + diff_logits.unsqueeze(1).repeat(1,C)*one_hot_indices) / tau
        y_soft = indexed_logits.softmax(dim=-1)

        if hard:
            y_soft_index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=\
                                  torch.legacy_contiguous_format).scatter_(dim,y_soft_index,1.0)
            out = y_hard - grad_coeff*y_soft.detach() + grad_coeff*y_soft
        else:
            out = y_soft
        return out


    def CRT(self, probs, index, CRT_coeff): # Categorical Reparameterization Trick
        # probs : Bn*T, C
        # index : Bn*T, 1
        BnT, C = probs.size()
        index = one_hot(index, self.class_tensor, C).squeeze()
        c = (index)*(1-CRT_coeff*probs.detach()) + (1-index)*(-CRT_coeff*probs.detach())
        return CRT_coeff*probs + c


    def sample_idx(self, probs, train, random_mode=False):
        if random_mode is True:
            return torch.randint(low=4,high=self.trg_words_n, size=(probs.size(0),1)).to(probs.device)

        gen_method = self.gen_method
        if self.sampling_ratio > 0.0:
            random_number = random.random()
            if random_number < self.sampling_ratio:
                if self.k > 1:
                    gen_method = 'topk_sampling'
                else:
                    gen_method = 'sampling'

        if train == False:
            gen_method = 'greedy'

        BnT, _ = probs.size()
        if gen_method == 'greedy':
            _, index = probs.topk(1, dim=1) # Bn, 1
        elif gen_method == 'sampling':
            m = torch.distributions.categorical.Categorical(probs)
            index = m.sample()
        elif gen_method == 'topk_sampling':
            topk_probs, topk_ids = probs.topk(self.k, dim=1) # best_probs : Bn, k
            topk_probs = topk_probs / torch.sum(topk_probs, dim=1).unsqueeze(1).repeat(1,self.k)
            m = torch.distributions.categorical.Categorical(topk_probs)
            tmp_index = m.sample()
            index = topk_ids[torch.arange(BnT), tmp_index.squeeze()].unsqueeze(1)
        return index

    def fast_generation_total(self, x_data, x_mask, y_data, y_mask, CRT_coeff):
        y_mask_in = y_mask.type(torch.float)

        y_out = y_data[:,1:]
        y_mask_in = y_mask_in[:,1:]
        y_in = y_data[:,:-1]

        Bn, Tx = x_data.size()
        enc_out = self.encoder(x_data, x_mask)

        pad = (torch.ones((Bn,1))*PAD).type(torch.long).cuda()
        pad_onehot = one_hot(pad, self.class_tensor, self.trg_words_n).squeeze() # Bn, C
        y_hat0 = (torch.ones((Bn,1))*BOS).type(torch.long).cuda()
        y_hat = one_hot(y_hat0, self.class_tensor, self.trg_words_n)

        dec_out = self.decoder(y_in, y_mask_in, enc_out, x_mask)
        dec_out = self.logit_layer(dec_out)
        Bn, T, C = dec_out.size()
        tmp_dec_out = dec_out.reshape(Bn*T, C)
        probs = F.softmax(tmp_dec_out, dim=1)

        if self.reparam_method == 'crt':
            tmp_y_hat = self.CRT(probs, y_out.reshape(-1,1), CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'gst':
            tmp_y_hat = self.org_gumbel_softmax(tmp_dec_out, tau=0.1, grad_coeff=CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'stgst':
            tmp_y_hat = self.org_gumbel_softmax(tmp_dec_out, tau=1.0, hard=True,\
                                                 grad_coeff=CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'igst':
            tmp_y_hat = self.index_gumbel_softmax(tmp_dec_out, y_out.reshape(-1,1),\
                                                    tau=0.1, grad_coeff=CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'istgst':
            tmp_y_hat = self.index_gumbel_softmax(tmp_dec_out, y_out.reshape(-1,1),\
                                                tau=1.0, hard=True, grad_coeff=CRT_coeff) # Bn*T, C
        tmp_y_mask = y_mask[:,1:].reshape(Bn*T, 1).repeat(1, C)
        tmp_y_hat = (tmp_y_mask)*tmp_y_hat + (1-tmp_y_mask)*pad_onehot.repeat(T,1)
        y_hat = torch.cat((y_hat, tmp_y_hat.reshape(Bn, T, C)), dim=1)

        return y_hat, y_mask, probs, enc_out


    def generation_total(self, x_data, x_mask, CRT_coeff, train=True):

        Bn, Tx = x_data.size()
        enc_out = self.encoder(x_data, x_mask)

        pad = (torch.ones((Bn,1))*PAD).type(torch.long).cuda()
        pad_onehot = one_hot(pad, self.class_tensor, self.trg_words_n).squeeze() # Bn, C

        EOSs = torch.zeros((Bn, 1)).cuda()

        y_hat0 = (torch.ones((Bn,1))*BOS).type(torch.long).cuda()
        dec_seq = copy.deepcopy(y_hat0)
        y_hat = one_hot(y_hat0, self.class_tensor, self.trg_words_n)
        dec_mask = CudaVariable(torch.ones((Bn,1))).type(torch.cuda.LongTensor)

        tmp_max_len = self.max_len-1
        for yi in range(tmp_max_len):

            dec_out = self.decoder(dec_seq, dec_mask, enc_out, x_mask) # Bn, Tx, C
            dec_out = self.logit_layer(dec_out)
            Bn, T, C = dec_out.size()
            tmp_dec_out = dec_out.reshape(Bn*T, C)
            probs = F.softmax(tmp_dec_out, dim=1) # Bn*T, C
            index = self.sample_idx(probs, train=train) # Bn*T, 1

            last_index = index.reshape(Bn, T, 1)[:,-1,:]
            tmp_EOSs = torch.gt(EOSs,0).type(torch.long)
            tmp_dec_seq = (1-tmp_EOSs)*last_index + (tmp_EOSs)*pad
            #dec_seq = torch.cat((dec_seq, last_index), dim=1)
            dec_seq = torch.cat((dec_seq, tmp_dec_seq), dim=1) # Bn, T+1
            dec_mask = torch.cat((dec_mask, (1-tmp_EOSs)), dim=1) # Bn, T+1

            #EOS1 = torch.eq(last_index, EOS).view(Bn, 1)
            EOS1 = torch.eq(tmp_dec_seq, EOS).view(Bn, 1)
            EOSs = EOSs + EOS1
            if yi > 0 and torch.sum(torch.gt(EOSs,0)) >= Bn:
                break

        EOSs = torch.gt(torch.eq(dec_seq, EOS).sum(dim=1),0).sum()
        if EOSs < Bn:
            add_dec = (torch.ones(Bn)*PAD).type(torch.long).cuda()
            add_mask = torch.zeros(Bn).type(torch.long).cuda()
            probs = probs.reshape(Bn, -1, C)
            add_probs = torch.zeros((Bn, C)).type(torch.float).cuda()
            add_probs[:,PAD] = 1.0
            for b in range(Bn):
                if torch.sum(torch.eq(dec_seq[b], EOS)) < 1:
                    add_dec[b] = EOS
                    add_mask[b] = 1
                    add_probs[b,PAD] = 0.0
                    add_probs[b,EOS] = 1.0
            dec_seq = torch.cat((dec_seq, add_dec.unsqueeze(1)), dim=1)
            dec_mask = torch.cat((dec_mask, add_mask.unsqueeze(1)), dim=1)
            probs = torch.cat((probs, add_probs.unsqueeze(1)), dim=1)
            probs = probs.reshape(-1,C)
            tmp_dec_out = torch.cat((tmp_dec_out.reshape(Bn,-1,C), add_probs.unsqueeze(1)), dim=1)
            tmp_dec_out = tmp_dec_out.reshape(-1,C)

            T += 1

        if self.reparam_method == 'crt':
            tmp_y_hat = self.CRT(probs, dec_seq[:,1:].reshape(-1,1), CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'gst':
            tmp_y_hat = self.org_gumbel_softmax(tmp_dec_out, tau=0.1, grad_coeff=CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'stgst':
            tmp_y_hat = self.org_gumbel_softmax(tmp_dec_out, tau=1.0, hard=True,\
                                                 grad_coeff=CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'igst':
            tmp_y_hat = self.index_gumbel_softmax(tmp_dec_out, dec_seq[:,1:].reshape(-1,1),\
                                                    tau=0.1, grad_coeff=CRT_coeff) # Bn*T, C
        elif self.reparam_method == 'istgst':
            tmp_y_hat = self.index_gumbel_softmax(tmp_dec_out, dec_seq[:,1:].reshape(-1,1),\
                                                tau=1.0, hard=True, grad_coeff=CRT_coeff) # Bn*T, C
        y_mask = copy.deepcopy(dec_mask) # Bn, T+1
        tmp_y_mask = y_mask[:,1:].reshape(Bn*T, 1).repeat(1, C) # Bn*T, C
        tmp_y_hat = (tmp_y_mask)*tmp_y_hat + (1-tmp_y_mask)*pad_onehot.repeat(T,1)
        y_hat = torch.cat((y_hat, tmp_y_hat.reshape(Bn, T, C)), dim=1) # Bn, T+1, C

        return y_hat, y_mask, probs, enc_out

class MyBTmodel(nn.Module):
    def __init__(self, args=None, LM=None):
        super(MyBTmodel, self).__init__()


        self.src_lang = args.src_lang
        self.trg_lang = args.trg_lang

        tmp_args = copy.deepcopy(args)

        tmp_args.src_words_n = args.src_words_n
        tmp_args.trg_words_n = args.trg_words_n
        self.xy_model = BT_Transformer(args=tmp_args)

        tmp_args.src_words_n = args.trg_words_n
        tmp_args.trg_words_n = args.src_words_n
        self.yx_model = BT_Transformer(args=tmp_args)

        if LM is not None:
            self.x_LM = LM.x_LM
            self.y_LM = LM.y_LM
        else:
            self.x_LM = None
            self.y_LM = None

        self.x_CRT_coeff = args.x_CRT_coeff
        self.y_CRT_coeff = args.y_CRT_coeff

        # Freezing the src, trg LMs
        if self.x_LM is not None:
            self.grad_setting(self.x_LM, False)
        if self.y_LM is not None:
            self.grad_setting(self.y_LM, False)

        self.test_max_len = 150

    def grad_setting(self, model, flag):
        for param in model.parameters():
            param.requires_grad = flag

    def alternate_freeze(self, mode):
        if mode == 'xyx_mono':
            self.grad_setting(self.xy_model.decoder, False)
            self.grad_setting(self.xy_model.logit_layer, False)
            #self.grad_setting(self.yx_model.encoder, False)
        elif mode == 'yxy_mono':
            self.grad_setting(self.yx_model.decoder, False)
            self.grad_setting(self.yx_model.logit_layer, False)
            #self.grad_setting(self.xy_model.encoder, False)
        elif mode == 'melt':
            self.grad_setting(self.xy_model, True)
            self.grad_setting(self.yx_model, True)

    def compute_LM_KL_loss(self, q_dist, lm_dist, mask, Bn, T):
        q_dist = q_dist.reshape(Bn, T, -1)
        log_q_dist = torch.log(q_dist+1e-20)
        log_lm_dist = torch.log(lm_dist+1e-20)

        mask = mask.type(torch.float)
        output = q_dist * (log_lm_dist - log_q_dist)
        output = torch.bmm(mask[:,1:].reshape(-1,1,1), output.reshape(Bn*T, 1, -1))
        output = -1 * output.reshape(Bn, T, -1)
        #output = torch.sum(output) / Bn
        return output

    def xyx_reconstruction(self, x_data, x_mask, load_y_data, load_y_mask):
        self.xy_model.eval()
        self.yx_model.train()
        if len(load_y_data) == 0:
            gen_y_data, gen_y_mask, q_dist, _ = self.xy_model.generation_total(x_data, x_mask,\
                                                                            self.y_CRT_coeff)
        else:
            gen_y_data, gen_y_mask, q_dist, _ = self.xy_model.fast_generation_total(\
                                        x_data, x_mask, load_y_data, load_y_mask, self.y_CRT_coeff)

        Bn, Ty, C = gen_y_data.size()
        y_enc_out = self.yx_model.encoder(gen_y_data, gen_y_mask)
        recon_loss = self.yx_model.decode_forward(y_enc_out, gen_y_mask, x_data, x_mask)
        recon_loss = torch.sum(recon_loss)/Bn

        if self.y_LM is not None:
            self.grad_setting(self.y_LM, False)
            self.y_LM.eval()
            _, lm_dist = self.y_LM.evaluate(gen_y_data.detach(), gen_y_mask) #- nll_loss
            LM_KL_loss = self.compute_LM_KL_loss(q_dist, lm_dist, gen_y_mask, Bn, Ty-1)
            LM_KL_loss = torch.sum(LM_KL_loss) / Bn
        else:
            LM_KL_loss = torch.zeros((Bn), dtype=torch.float).cuda()
            LM_KL_loss = torch.sum(LM_KL_loss) / Bn
        LM_KL_loss += 0*recon_loss
        return recon_loss, LM_KL_loss, gen_y_data, gen_y_mask

    def yxy_reconstruction(self, y_data, y_mask, load_x_data, load_x_mask):
        self.yx_model.eval()
        self.xy_model.train()
        if len(load_x_data) == 0:
            gen_x_data, gen_x_mask, q_dist, _ = self.yx_model.generation_total(y_data, y_mask,\
                                                                            self.x_CRT_coeff)
        else:
            gen_x_data, gen_x_mask, q_dist, _ = self.yx_model.fast_generation_total(\
                                       y_data, y_mask, load_x_data, load_x_mask, self.x_CRT_coeff)
        Bn, Tx, C = gen_x_data.size()
        x_enc_out = self.xy_model.encoder(gen_x_data, gen_x_mask)
        recon_loss = self.xy_model.decode_forward(x_enc_out, gen_x_mask, y_data, y_mask)
        recon_loss = torch.sum(recon_loss)/Bn

        if self.x_LM is not None:
            self.grad_setting(self.x_LM, False)
            self.x_LM.eval()
            _, lm_dist = self.x_LM.evaluate(gen_x_data.detach(), gen_x_mask) #- nll_loss
            LM_KL_loss = self.compute_LM_KL_loss(q_dist, lm_dist, gen_x_mask, Bn, Tx-1)
            LM_KL_loss = torch.sum(LM_KL_loss) / Bn
        else:
            LM_KL_loss = torch.zeros((Bn), dtype=torch.float).cuda()
            LM_KL_loss = torch.sum(LM_KL_loss) / Bn
        LM_KL_loss += 0*recon_loss
        return recon_loss, LM_KL_loss, gen_x_data, gen_x_mask

    def xy_translation(self, x_data, x_mask, y_data, y_mask):
        Bn, Tx = x_data.size()
        x_enc_out = self.xy_model.encoder(x_data, x_mask)

        trans_loss = self.xy_model.decode_forward(x_enc_out, x_mask, y_data, y_mask)
        trans_loss = torch.sum(trans_loss) / Bn
        return trans_loss, x_enc_out

    def yx_translation(self, x_data, x_mask, y_data, y_mask):
        Bn, Ty = y_data.size()
        y_enc_out = self.yx_model.encoder(y_data, y_mask)

        trans_loss = self.yx_model.decode_forward(y_enc_out, y_mask, x_data, x_mask)
        trans_loss = torch.sum(trans_loss) / Bn
        return trans_loss, y_enc_out

    def x_generation(self, y_data, y_mask, multi_gpu=False, B_max=0, max_len=0):
        start_time = time.time()
        y_enc_out = self.yx_model.encoder(y_data, y_mask)

        gen_x_data, gen_x_mask, _, times =\
                     self.yx_model.translate(y_enc_out, y_mask, 1.0,\
                                                         train=False, start_time=start_time)
        return gen_x_data.detach().cpu().numpy(), y_enc_out, times

    def y_generation(self, x_data, x_mask, multi_gpu=False, B_max=0, max_len=0):
        start_time = time.time()
        x_enc_out = self.xy_model.encoder(x_data, x_mask)

        gen_y_data, gen_y_mask, _, times =\
                     self.xy_model.translate(x_enc_out, x_mask, 1.0,\
                                                         train=False, start_time=start_time)
        return gen_y_data.detach().cpu().numpy(), x_enc_out, times

    def forward(self, *inputs, **kwargs):
        mode = inputs[-1]
        if mode == 'bilingual_process':
            x_data, x_mask, y_data, y_mask = inputs[0],inputs[1],inputs[2],inputs[3]

            x_data = CudaVariable(torch.LongTensor(x_data))
            x_mask = CudaVariable(torch.LongTensor(x_mask))
            y_data = CudaVariable(torch.LongTensor(y_data))
            y_mask = CudaVariable(torch.LongTensor(y_mask))

            B, _ = x_data.size()

            xy_trans_loss, _ = self.xy_translation(x_data, x_mask, y_data, y_mask)
            yx_trans_loss, _ = self.yx_translation(x_data, x_mask, y_data, y_mask)
            return (xy_trans_loss, yx_trans_loss)
        elif mode == 'xyx_monolingual_process':
            x_data, x_mask, load_y_data, load_y_mask = inputs[0],inputs[1],inputs[2],inputs[3]

            x_data = CudaVariable(torch.LongTensor(x_data))
            x_mask = CudaVariable(torch.LongTensor(x_mask))
            load_y_data = CudaVariable(torch.LongTensor(load_y_data))
            load_y_mask = CudaVariable(torch.LongTensor(load_y_mask))

            recon_loss, LM_KL_loss, gen_y_data, gen_y_mask = self.xyx_reconstruction(\
                                                x_data, x_mask, load_y_data, load_y_mask)
            return (recon_loss, LM_KL_loss, gen_y_data, gen_y_mask)
        elif mode == 'yxy_monolingual_process':
            y_data, y_mask, load_x_data, load_x_mask = inputs[0],inputs[1],inputs[2],inputs[3]

            load_x_data = CudaVariable(torch.LongTensor(load_x_data))
            load_x_mask = CudaVariable(torch.LongTensor(load_x_mask))
            y_data = CudaVariable(torch.LongTensor(y_data))
            y_mask = CudaVariable(torch.LongTensor(y_mask))

            recon_loss, LM_KL_loss, gen_x_data, gen_x_mask = self.yxy_reconstruction(\
                                                y_data, y_mask, load_x_data, load_x_mask)
            return (recon_loss, LM_KL_loss, gen_x_data, gen_x_mask)

        elif mode == 'x_generation':
            y_data, y_mask = inputs[0],inputs[1]

            y_data = CudaVariable(torch.LongTensor(y_data))
            y_mask = CudaVariable(torch.LongTensor(y_mask))

            gen_x_data, _, _ = self.x_generation(y_data, y_mask)
            gen_x_data = gen_x_data[:,1:] # deleting BOS token

            return (torch.from_numpy(gen_x_data).cuda())
        elif mode == 'y_generation':
            x_data, x_mask = inputs[0],inputs[1]

            x_data = CudaVariable(torch.LongTensor(x_data))
            x_mask = CudaVariable(torch.LongTensor(x_mask))

            gen_y_data, _, _ = self.y_generation(x_data, x_mask)
            gen_y_data = gen_y_data[:,1:] # deleting BOS token

            return (torch.from_numpy(gen_y_data).cuda())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--src_lang", type=str, default='en')
    parser.add_argument("--trg_lang", type=str, default='de')

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--model", type=str, default='nmt')
    parser.add_argument("--joined_dictionary", type=int, default=1)
    parser.add_argument("--dim_wemb", type=int, default=256)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--emb_noise", type=float, default=0.0)
    parser.add_argument("--tm_n_layers", type=int, default=3)
    parser.add_argument("--tm_dim_ff", type=int, default=1024)
    parser.add_argument("--tm_n_head", type=int, default=4)
    parser.add_argument("--tm_dk", type=int, default=64)
    parser.add_argument("--tm_dv", type=int, default=64)
    parser.add_argument("--tm_resnorm_type", type=str, default='norm_res')
    parser.add_argument("--gen_method", type=str, default='greedy')
    parser.add_argument("--sampling_ratio", type=float, default=0.0)
    parser.add_argument("--sampling_annealing", type=int, default=0)
    parser.add_argument("--n_topk_sampling", type=int, default=1) # only for sampling gen_method
    parser.add_argument("--x_CRT_coeff", type=float, default=0.001)
    parser.add_argument("--y_CRT_coeff", type=float, default=0.001)
    parser.add_argument("--x_LM_kl_loss_coeff", type=float, default=0.001)
    parser.add_argument("--y_LM_kl_loss_coeff", type=float, default=0.001)

    args = parser.parse_args()

    B = 32
    T = 32
    V = 10000
    args.src_words_n = V
    args.trg_words_n = V

    model = MyBTmodel(args=args)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def backward_update(model, loss, mode='bi'):

        model.alternate_freeze(mode)
        loss.backward()
        model.alternate_freeze('melt')

    bi_x_data = torch.randint(V, (B, T))
    bi_x_mask = torch.ones_like(bi_x_data)
    bi_y_data = torch.randint(V, (B, T))
    bi_y_mask = torch.ones_like(bi_y_data)
    mono_x_data = torch.randint(V, (B, T))
    mono_x_mask = torch.ones_like(mono_x_data)
    mono_y_data = torch.randint(V, (B, T))
    mono_y_mask = torch.ones_like(mono_y_data)

    load_x_data = []
    load_x_mask = []
    load_y_data = []
    load_y_mask = []

    # Bilingual Process
    model.train()
    (xy_loss, yx_loss) = model(bi_x_data, bi_x_mask, bi_y_data, bi_y_mask, 'bilingual_process')
    bi_loss = xy_loss + yx_loss

    # xyx Monolignaul Process
    model.train()
    (recon_loss, LM_kl_loss, gen_y_data, gen_y_mask) =\
           model(mono_x_data, mono_x_mask, load_y_data, load_y_mask, 'xyx_monolingual_process')
    xyx_loss = recon_loss + args.y_LM_kl_loss_coeff*LM_kl_loss

    # Trg Monolingual Process
    model.train()

    (recon_loss, LM_kl_loss, gen_x_data, gen_x_mask) =\
           model(mono_y_data, mono_y_mask, load_x_data, load_x_mask, 'yxy_monolingual_process')
    yxy_loss = recon_loss + args.x_LM_kl_loss_coeff*LM_kl_loss

    model.zero_grad()
    optimizer.zero_grad()

    backward_update(model, bi_loss, mode='bi')
    backward_update(model, xyx_loss, mode='xyx_mono')
    backward_update(model, yxy_loss, mode='yxy_mono')

    if args.grad_clip > 0.0:
        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
    optimizer.step()

    print("A training iteration is done")
