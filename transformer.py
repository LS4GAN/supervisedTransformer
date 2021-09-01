#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math, copy, time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
seaborn.set_context(context="talk")


# Model Architecture
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and mnay other models
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        # we have to first embed the source and target
        # and send the embedded source to encoder and
        # embedded target to the decoder.
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # generator converts the output embedding 
        # from the decoder to vocabulary
        self.generator = generator

    def forward(self, src, tgt):
        """
        Take in and process masked source and target sequences.
        """
        # MORE ANNOTATION HERE, ESPECIALLY FOR THE MASKS
        memory = self.encode(src)
#         print(memory.shape)
        output = self.decode(memory, tgt)
#         print(output.shape)
#         input()
        return self.generator(output)

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, memory, tgt):
        return self.decoder(self.tgt_embed(tgt), memory)

# class Embeddings(nn.Module):
#     def __init__(self, d_model, vocab):
#         """
#         Input:
#             - d_model (int): the embedding dimension
#             - vocab (int): size of the vocabulary. (Yi) we have to think about this part, we don't have a fixed sized vocabulary
#         """
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(vocab, d_model)
#         self.d_model = d_model

#     def forward(self, x):
#         # x = x.type(torch.LongTensor)
#         return self.lut(x) * math.sqrt(self.d_model)

# class Generator(nn.Module):
#     """
#     Define standard linear + softmax generation step.
#     """
#     def __init__(self, d_model, vocab):
#         """
#         d_model: MORE ANNOTATION HERE
#         vocab: MORE ANNOTATION HERE
#         """
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(d_model, vocab)

#     def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        """
        Construct a layernorm module
        Input:
            - size (int): The size of the last dimension of the input (x) to forward. 
                Since a_2 and b_2 have to be trainable parameters, we have to pass size
                at initialization instead of getting it from x.
            - eps (float): added to the denominator (std) to avoid dividing by zero.
        """
        super(LayerNorm, self).__init__()
        # a_2 and b_2 are parameters that are trainable
        # nn.Parameter(tensor, required_grad=True)
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        """
        Calculate the mean and std along the last dimension, 
        and keep the dimension (e.g. 3x5x2 to 3x5x1, instead of 3x5)
        NOTE: for two TENSORS to be addable, they must match on the last dimension.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)
        return  x_normalized * self.a_2 + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    NOTE 1: For code simplicity, we do norm first as opposed to last.
    NOTE 2: (Yi) ViTGAN followed this norm-first approach
    """
    def __init__(self, size, dropout):
        """
        Input:
            - size: the embedding dimension of a word (or token);
            - dropout (float): probability of an element to be zeroed (default: .5).
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))

def subsequent_mask(size):
    """
    Mask out subsequence positions
    """
    attn_shape = (1, size, size)
    # make a upper triangular tensor
    # k is the offset from diagonal
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# Encoder
class EncoderLayer(nn.Module):
    """
    Encoder is made up of two major blocks: 
        - self-attention, and
        - feed forward.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Input:
            - size: the embedding dimension of a word (or token);
            - self_attn: self-attention layers
            - feed_forward: feed-forward layers
            - dropout (float): probability of an element to be zeroed (default: .5)
        """
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        

    def forward(self, x, mask):
        """
        Input:
            - x (tensor): the Key (K), Value (V), and Query (Q) matrix. For self-attention, we can have all three of them identical.
            - mask: We use matrices K and Q to generate scores. 
                However, we can only allow a token to attend to tokens generated no latter than itself. 
                Hence, we have to mask the score vector so that only values in values after i are zerod out in the softmax
        """
        # the self attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # the feed forward sublayer
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):    
    def __init__(self, layer, N):
        """
        Core encoder is a stack of N (default=6) layers.
        input:
            - layer (class nn.Module): the transformer encoder block
            - N (int): the number of transformer encoder blocks
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        """
        Pass the input (and mask) through each layer in turn.
        Input:
            - x (tensor): the input tnesor
            - mask (function): the mask function to attention
        """
        for layer in self.layers:
            x = layer(x, None)
        return self.norm(x)


# Decoder
class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    """
    Generic N-layered decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory):
#         print(f'target size = {x.shape}')
#         input()
        seq_len = x.size(1)
        tgt_mask = Variable(subsequent_mask(seq_len))
        if x.is_cuda:
            tgt_mask = tgt_mask.cuda()
        for layer in self.layers:
            x = layer(x, memory, None, tgt_mask)
        return self.norm(x)

# Attention
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute the scaled dot-product attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask_fill(mask, value_to_fill)
        # since we are using softmax, fill the score with a really negative number will make the probability close to zero
        # NOTE: About the mask
        # - mask and the tensor to be mask MUST have the same number of dimensions, i.e. mask and tensor must both be n-dimensional
        # - Suppose the tensor is of shape (3, 4, 2, 8)
        #   then the mask can have shape (3, 1, 1, 8)
        #   The requirement is that the size of mask must match the size of tensor at all non-singleton.
        #   That is, we cannot have mask of shape (3, 2, 1, 8) for tensor of shape (3, 4, 2, 8).
        #   We broadcast behavior on non-singleton dimension to the singleton dimension(s).
        #   This explains why we generate source masking and target masking in the different way.
        #   Source masking actually do not mask anything;
        #   But target masking need to mask all words come after.

        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=.1):
        """
        Take in model size and number of head
        Input:
            - h (int): number of attention heads;
            - d_model (int): dimension of the output from the model
            - dropout (float): 
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be a multiple of the number of heads (h)"
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concatenate using a vew and apply a final linear
        # contiguous() makes a copy of the tensor whose memory layout is confirmed to its shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# Position-wise Feed-Forward Networks
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
        """
        Input:
            - d_model: the embedding length of each token in the sequence
            - dropout: after adding the positional encoding, we apply dropout to the sum tensor
            - max_len: buffer length. 
                NOTE: Make sure it is LONGER than the number of tokens in your sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # make a max_len x 1 tensor where the [i, 0] entry equaling i
        div_term = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model) # 1 / denominator 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# # Assemble the component and make a model
# def make_model(
#     src_vocab, 
#     tgt_vocab, 
#     N=6, 
#     d_model=512, 
#     d_ff=2048, 
#     h=8, 
#     dropout=.1):
#     """
#     Construct a model from hyperparameters
#     """

#     C = copy.deepcopy

#     attn = MultiHeadedAttention(h, d_model)
#     ff = PositionWiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)

#     encoder = Encoder(EncoderLayer(d_model, C(attn), C(ff), dropout), N)
#     decoder = Decoder(DecoderLayer(d_model, C(attn), C(attn), C(ff), dropout), N)
#     model = EncoderDecoder(
#         encoder,
#         decoder,
#         nn.Sequential(Embeddings(d_model, src_vocab), C(position)),
#         nn.Sequential(Embeddings(d_model, tgt_vocab), C(position)),
#         Generator(d_model, tgt_vocab)
#     )

#     # (author of this blog): This was important from their code.
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model
