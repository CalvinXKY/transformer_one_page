#  A simple way of transformer model.
#  Original Paper: https://arxiv.org/pdf/1706.03762.pdf
#  *  Author: kevin.xie
#  *  Email: kaiyuanxie@yeah.net

import copy
import time
import torch
import torch.nn as nn

"""
ABBR.
bs: batch size,
seq_len: max src/trg token-sequence length,
dk: key/value size; head dimensionality
heads/h: number of heads
d_model: model dimension
pe: positional encoding
dff:  inner-layer dimensionality
p_drop: probability of dropout
ffn:  position-wise feed-forward networks
MHA: multi-head attention
"""


def replicate_module(module, copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(copies)])

# Part1: ================== modules ==================


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.dk = d_model // heads  # head dimension
        self.heads = heads
        self.qkv_nets = (nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model))
        self.out_linear = nn.Linear(d_model, d_model)
        self.sqrt_dk = torch.sqrt(torch.tensor(self.dk))

    # Scaled dot-product attention:
    def attention(self, query, key, value, mask):
        # query/key/value shape (bs, heads, seq_len, dk)
        # mask shape = (bs, 1, 1, seq_len) or (bs, 1, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dk  # shape: (bs, heads, seq_len, seq_len)
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))
        # Softmax dim=-1 stands for apply the softmax along the last dimension
        attention_weights = nn.Softmax(dim=-1)(scores)  # shape: (bs, heads, seq_len, seq_len)
        attention_qkv = torch.matmul(attention_weights, value)   # shape:  (bs, heads, seq_len, dk)
        return attention_qkv

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        # qkv shape: (bs, seq_len, dk*heads)
        # dk * heads = d_model
        query, key, value = [net(x).view(batch_size, -1, self.heads, self.dk).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        attention_qkv = self.attention(query, key, value, mask)  # shape:  (bs, heads, seq_len, dk)
        #  (bs, heads, seq_len, dk) -> (bs, seq_len, dk*heads)
        reshaped = attention_qkv.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dk)
        representations_batch = self.out_linear(reshaped)
        return representations_batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=None, max_seq_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop) if p_drop is not None else None
        position_id = torch.arange(0, max_seq_length).unsqueeze(1)  # (max_seq_length, 1)
        frequencies = torch.pow(10000., -torch.arange(0, d_model, 2, dtype=torch.float) / d_model)

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        pe[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions
        self.register_buffer('pe', pe)

    def forward(self, embeddings_batch):
        # embedding_batch  shape: (bs, seq_len, d_model)
        # pe shape: (max_seq_length, d_model)
        # pe shape broad_casted -> (bs, seq_len, d_model)
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.pe.shape[-1]
        positional_encodings = embeddings_batch + self.pe[:embeddings_batch.shape[1]]
        if self.dropout is not None:
            positional_encodings = self.dropout(positional_encodings)
        return positional_encodings


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embeddings_layer = nn.Embedding(vocab_size, d_model)
        self.sqrt_d_model = torch.sqrt(torch.tensor(d_model))

    def forward(self, tokens):
        assert tokens.ndim == 2
        # tokens shape: (bs, seq_len)
        # embeddings shape: (bs, seq_len, d_model), every token id has associated vector
        embeddings = self.embeddings_layer(tokens)
        # Paper P-5, Chapter 3.4 "Embeddings and Softmax": multiply the embedding weights by the square root of d_model
        embeddings = embeddings * self.sqrt_d_model
        return embeddings


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dff=1024):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.relu(self.linear1(representations_batch)))


class AddNormLayer(nn.Module):
    def __init__(self, d_model, p_prob):
        super().__init__()
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_prob)

    def forward(self, representations_batch, sublayer_module):
        return representations_batch + self.dropout(sublayer_module(self.LN(representations_batch)))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, p_prob,):
        super().__init__()
        self.sublayers = replicate_module(AddNormLayer(d_model, p_prob), 2)
        self.multi_headed_attention = MultiHeadedAttention(d_model, heads)
        self.ffn = PositionwiseFeedForward(d_model)

        self.d_model = d_model

    def forward(self, src_representations_batch, src_mask):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.ffn)

        return src_representations_batch


class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads, p_prob):
        super().__init__()
        self.sublayers = replicate_module(AddNormLayer(d_model, p_prob), 3)
        self.trg_multi_headed_attention = MultiHeadedAttention(d_model, heads)
        self.src_multi_headed_attention = MultiHeadedAttention(d_model, heads)
        self.ffn = PositionwiseFeedForward(d_model)
        self.d_model = d_model

    def forward(self, trg_representations_batch, src_representations_batch, trg_mask, src_mask):
        srb = src_representations_batch
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)
        decoder_src_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by a source-attending MHA and point-wise feed forward net sublayer
        trg_representations_batch = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)
        trg_representations_batch = self.sublayers[1](trg_representations_batch, decoder_src_attention)
        trg_representations_batch = self.sublayers[2](trg_representations_batch, self.ffn)

        return trg_representations_batch


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, trg_representations_batch):
        # trg_representations_batch shape: (bs, seq_len, d_model)
        # output shape: (bs, seq_len, vocab_size)
        return self.log_softmax(self.linear(trg_representations_batch))

# Part2: =================== Encoder&Decoder ======================


class Encoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        self.encoder_layers = replicate_module(encoder_layer, number_of_layers)
        self.LN = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src_embeddings_batch, src_mask):
        src_representations_batch = src_embeddings_batch
        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)
        return self.LN(src_representations_batch) # Using LN. not mentioned explicitly in the paper.


class Decoder(nn.Module):
    def __init__(self, decoder_layer, number_of_layers):
        super().__init__()
        self.decoder_layers = replicate_module(decoder_layer, number_of_layers)
        self.LN = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, trg_embeddings_batch, src_representations_batch, trg_mask, src_mask):
        trg_representations_batch = trg_embeddings_batch

        # Forward pass through the decoder stack
        for decoder_layer in self.decoder_layers:
            trg_representations_batch = decoder_layer(trg_representations_batch, src_representations_batch,
                                                      trg_mask, src_mask)
        return self.LN(trg_representations_batch)  # Using LN. not mentioned explicitly in the paper.

# Part3: ================== transformer ==================


class Transformer(nn.Module):
    def __init__(self, d_model, src_vocab_size, trg_vocab_size, heads, number_of_layers, p_prob):
        super().__init__()

        # Embeds source/target token ids into embedding vectors
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.trg_embedding = Embedding(trg_vocab_size, d_model)

        # Adds positional information to source/target token's embedding vector
        # (otherwise we'd lose the positional information which is important in human languages)
        self.src_pos_embedding = PositionalEncoding(d_model, p_prob)
        self.trg_pos_embedding = PositionalEncoding(d_model, p_prob)

        encoder_layer = EncoderLayer(d_model, heads, p_prob)
        decoder_layer = DecoderLayer(d_model, heads, p_prob)

        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.decoder = Decoder(decoder_layer, number_of_layers)

        # Converts final target token representations into log probabilities vectors of the target vocab size
        self.generator = Generator(d_model, trg_vocab_size)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, src_token_ids_batch, trg_token_ids_batch, src_mask, trg_mask):
        src_representations_batch = self.encode(src_token_ids_batch, src_mask)
        trg_log_probs = self.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)
        return trg_log_probs

    def encode(self, src_token_ids_batch, src_mask):
        src_embeddings_batch = self.src_embedding(src_token_ids_batch)  # (bs, seq_len) -> (bs, seq_len, d_model)
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)

        return src_representations_batch

    def decode(self, trg_token_ids_batch, src_representations_batch, trg_mask, src_mask):
        trg_embeddings_batch = self.trg_embedding(trg_token_ids_batch)  # (bs, seq_len) -> (bs, seq_len, d_model)
        trg_embeddings_batch = self.trg_pos_embedding(trg_embeddings_batch)
        trg_representations_batch = self.decoder(trg_embeddings_batch, src_representations_batch, trg_mask, src_mask)

        # linear projection followed by log softmax
        trg_log_probs = self.generator(trg_representations_batch) # (bs, seq_len, d_model) -> (bs, seq_len, vocab_size)

        # (bs*seq_len, vocab_size) format for passing it into KL div loss
        trg_log_probs = trg_log_probs.reshape(-1, trg_log_probs.shape[-1]) # (bs, seq_len, vocab_size) -> (bs*seq_len, vocab_size)

        return trg_log_probs


# Part4: ==================== tests =======================
def time_print(foo, interval=2):
  def func(*args, **kwargs):
    print("\n")
    result = foo(*args,**kwargs)
    time.sleep(interval)
    print("=+"*30, "\n")
    return result
  return func


@time_print
def test_multi_head_attention():
    bs = 4
    seq_len = 1024
    d_model = 512
    test_qkv = torch.ones(bs, seq_len, d_model)
    multi_headed_attention = MultiHeadedAttention(d_model=d_model, heads=8)
    output = multi_headed_attention(test_qkv, test_qkv, test_qkv, None)
    print(f"Test multi_head_attention. Input shape:{(bs, seq_len, d_model)} Output shape: {output.shape}")
    assert output.shape == (bs, seq_len, d_model)


@time_print
def test_positional_encoding():
    bs = 2
    seq_len = 16
    d_model = 4
    embeddings_batch = torch.zeros(bs, seq_len, d_model)
    pe = PositionalEncoding(d_model,  max_seq_length=128)
    output = pe(embeddings_batch)
    print("Test positional_encoding. PE value:\n", output)


@time_print
def test_transformer():
    vocab_size = 1024
    batch_size = 4
    seq_length = 100
    transformer = Transformer(d_model=512, src_vocab_size=vocab_size, trg_vocab_size=vocab_size,
                              heads=8, number_of_layers=2, p_prob=0.3)
    src_token_ids_batch = torch.randint(0, 1000, size=(batch_size, seq_length))
    trg_token_ids_batch = torch.randint(0, 1000, size=(batch_size, seq_length))
    out = transformer(src_token_ids_batch, trg_token_ids_batch, src_mask=None, trg_mask=None)
    print(f"Test transformer. Input shape: {(batch_size, seq_length)} Output shape: {out.shape}")
    assert out.shape == (batch_size*seq_length, vocab_size)


if __name__ == "__main__":
    test_multi_head_attention()
    test_positional_encoding()
    test_transformer()
