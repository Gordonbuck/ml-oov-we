import torch
import torch.nn as nn


class PositionAttentionEncoding(nn.Module):
    def __init__(self, n_hid, n_seq, dropout=0.1, max_len=1000):
        super(PositionAttentionEncoding, self).__init__()
        self.pos_att = nn.Parameter(torch.ones(n_seq))
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros((max_len, n_hid))
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid, 2.) / n_hid))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe / torch.sqrt(torch.tensor(n_hid))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = (x.transpose(-2, -1) * self.pos_att).transpose(-2, -1)
        return self.dropout(x + self.pe[:x.shape[-2]])


class ResidualConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_hid, n_head, d, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d = d
        self.n_head = n_head
        self.q, self.k, self.v = (nn.Linear(n_hid, n_hid) for _ in range(3))
        self.out = nn.Linear(d * n_head, n_hid)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        n_batches = x.size(0)

        query = self.q(x).view(n_batches, -1, self.n_head, self.d).transpose(1, 2)
        key = self.k(x).view(n_batches, -1, self.n_head, self.d).transpose(1, 2)
        value = self.v(x).view(n_batches, -1, self.n_head, self.d).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim=-1)
        x = self.dropout(torch.matmul(p_attn, value))

        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_head * self.d)
        return self.out(x)


class PositionFeedForward(nn.Module):
    def __init__(self, n_hid, dropout=0.1):
        super(PositionFeedForward, self).__init__()
        self.w_1 = nn.Linear(n_hid, n_hid * 2)
        self.w_2 = nn.Linear(n_hid * 2, n_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))


class SelfAttentionFFN(nn.Module):
    def __init__(self, n_head, n_hid, att_d, att_dropout=0.1, ffn_dropout=0.1, res_dropout=0.3):
        super(SelfAttentionFFN, self).__init__()
        self.self_attn = MultiHeadedAttention(n_hid, n_head, att_d, att_dropout)
        self.feed_forward = PositionFeedForward(n_hid, ffn_dropout)
        self.res_1 = ResidualConnection(n_hid, res_dropout)
        self.res_2 = ResidualConnection(n_hid, res_dropout)

    def forward(self, x, mask=None):
        x = self.res_1(x, lambda y: self.self_attn(y, mask=mask))
        return self.res_2(x, self.feed_forward)


class CharacterCNN(nn.Module):
    def __init__(self, n_hid, dropout=0.3):
        super(CharacterCNN, self).__init__()
        self.char_emb = nn.Embedding(26 + 1, n_hid)
        self.filter_num_width = [2, 4, 6, 8]
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=n_hid, out_channels=n_hid, kernel_size=filter_width),
                nn.ReLU()
            ) for filter_width in self.filter_num_width])
        self.linear = nn.Linear(n_hid * len(self.filter_num_width), n_hid)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_hid, eps=1e-6)

    def forward(self, x):
        x = self.char_emb(x).transpose(-1, -2)  # B * H * W
        conv_out = [torch.max(conv(x), dim=-1)[0] for conv in self.convs]
        conv_out = self.dropout(torch.cat(conv_out, dim=-1))
        return self.norm(self.linear(conv_out))  # B * H


class HICE(nn.Module):
    def __init__(self, n_head, n_hid, n_seq, n_layer, idx2vec, use_morph=True, emb_tunable=False):
        super(HICE, self).__init__()
        self.n_hid = n_hid  # H
        self.n_seq = n_seq  # L
        self.n_layer = n_layer
        self.use_morph = use_morph
        self.emb_tunable = emb_tunable
        self.emb = nn.Embedding(len(idx2vec), n_hid)
        self.update_embedding(idx2vec, init=True)
        self.pos_att = PositionAttentionEncoding(n_hid, n_seq)
        self.ce_sa_layers = nn.ModuleList([SelfAttentionFFN(n_head, n_hid, n_hid // n_head) for _ in range(n_layer)])
        self.mca_sa_layers = nn.ModuleList([SelfAttentionFFN(n_head, n_hid, n_hid // n_head) for _ in range(n_layer)])
        if use_morph:
            self.char_cnn = CharacterCNN(n_hid)
        self.out = nn.Linear(n_hid, n_hid)
        self.bal = nn.Parameter(torch.ones(2) / 10.)

    def update_embedding(self, w2v, init=False):
        target_w2v = torch.tensor(w2v)
        if not init:
            origin_w2v = self.emb.weight
            target_w2v[:origin_w2v.shape[0]] = origin_w2v
        self.emb.weight = nn.Parameter(target_w2v)
        self.emb.weight.requires_grad = self.emb_tunable

    def mask_pad(self, x, pad=0):
        return (x != pad).unsqueeze(-2).unsqueeze(-2)

    def get_bal(self, n_cxt):
        # shorter the context length, the higher we should rely on morphology.
        return torch.sigmoid(self.bal[0] * n_cxt + self.bal[1])

    def forward(self, contexts, chars=None, pad=0):
        # contexts : B (batch size) * K (num contexts) * L (max num words in context) : contains word indices
        # vocabs : B (batch size) * W (max number of characters in target words) : contains character indices
        masks = self.mask_pad(contexts, pad).transpose(0, 1).float()  # K * B * L
        x = self.pos_att(self.emb(contexts)).transpose(0, 1)  # K * B * L * H (word emb size)

        # apply SA and FFN to each context, then average over words for each context
        res = []
        for xi, mask in zip(x, masks):
            for layer in self.mca_sa_layers:
                xi = layer(xi, mask=mask)
            res += [torch.sum(xi * mask, dim=1) / torch.sum(mask, dim=1)]
        res = torch.stack(res).transpose(0, 1)  # B * K * H

        # apply SA and FFN to aggregated context
        for layer in self.context_aggegator:
            res = layer(res)

        # weighted average with character CNN
        if self.use_morph and not (chars is None):
            cxt_weight = self.get_bal(contexts.shape[-1])
            res = cxt_weight * res.mean(dim=1) + (1. - cxt_weight) * self.char_cnn(chars)
        else:
            res = res.mean(dim=1)  # B * H

        return self.out(res)
