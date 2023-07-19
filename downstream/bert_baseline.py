from turtle import distance
import torch
import torch.nn as nn
import numpy as np
import math


'''
relate position embedding
'''

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position,args):
        super().__init__()
        self.device = 'cuda:%s' % str(args.gpu)
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

        self.zero = nn.Parameter(torch.zeros((num_units)))[None, :].to(self.device)
        #

    def forward(self, length_q, length_k,relative_v):
        if relative_v == 2:  # 版本2初始化
            range_vec_q = torch.arange(length_q - 2)
            range_vec_k = torch.arange(length_k - 2)
        else:  # 版本0 ，1 初始化 
            range_vec_q = torch.arange(length_q)
            range_vec_k = torch.arange(length_k)

        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]

        if relative_v == 2:
            distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position - 1,
                                               self.max_relative_position + 1)
        else:
            distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)

        # padding
        if relative_v == 0:  # 不padding的时候
            pass
        elif relative_v == 1:  # 开头结尾当作正常值做padding
            pa = torch.zeros((length_q, length_k))
            for i in range(self.max_relative_position, 0, -1):
                for j in range(i, 0, -1):
                    pa[self.max_relative_position - i][-j] = int(2 * self.max_relative_position + j - i)
            pa += pa.permute(1, 0) * -1
            distance_mat_clipped = distance_mat_clipped - pa
            distance_mat_clipped = distance_mat_clipped.type(torch.int64)

        elif relative_v == 2:  # 不考虑开头结尾，并且clip掉大于k的部分
            pa = torch.zeros((length_q - 2, length_k - 2))
            for i in range(self.max_relative_position, 0, -1):
                for j in range(i, 0, -1):
                    pa[self.max_relative_position - i][-j] = int(2 * self.max_relative_position + j - i) + 1
            pa += pa.permute(1, 0) * -1
            distance_mat_clipped = torch.nn.functional.pad(distance_mat_clipped - pa, (1, 1, 1, 1), "constant", 0)
            distance_mat_clipped = distance_mat_clipped.type(torch.int64)
            embeddings_table_pad = torch.cat((self.zero, self.embeddings_table, self.zero), 0)
        # padding
        if relative_v == 2:
            final_mat = distance_mat_clipped + (self.max_relative_position + 1)
        else:
            final_mat = distance_mat_clipped + self.max_relative_position

        final_mat = torch.LongTensor(final_mat).to(self.device)
        if relative_v == 2:
            embeddings = embeddings_table_pad[final_mat].to(self.device)
        else:
            embeddings = self.embeddings_table[final_mat].to(self.device)

        return embeddings

'''
relate position embedding
'''


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_size):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        ans = self.pe[:, :x.size(1)]
        return self.pe[:, :x.size(1)]


class Embedding(nn.Module):
    def __init__(self, args,vocab_size):
        max_len = args.seq_len # 
        super(Embedding, self).__init__()
        self.args = args
        if self.args.is_training ==2:
            self.tok_embed = nn.Embedding(vocab_size, args.d_model)  # token embedding
            if self.args.word2vec_grad:
                for param in self.tok_embed.parameters():
                        param.requires_grad = True
            else:
                for param in self.tok_embed.parameters():
                        param.requires_grad = False
        else:
            self.tok_embed = nn.Embedding(vocab_size, args.d_model)  # token embedding
        if args.if_posiemb == 1 or args.if_posiemb == 3:
            self.pos_embed = PositionalEncoding(max_len, args.d_model)
        elif args.if_posiemb == 2 :
            self.pos_embed = nn.Embedding(max_len, args.d_model)

        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x, user=None, temporal=None):
        device = 'cuda:%s' % str(self.args.gpu)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(device)  # [seq_len] -> [batch_size, seq_len]
        if self.args.if_posiemb:
            embedding = self.tok_embed(x) + self.pos_embed(pos)
        else:
            embedding = self.tok_embed(x)

        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self,args):
        super(ScaledDotProductAttention, self).__init__()
        self.args = args
    def forward(self, Q, K, V, attn_mask, a=None, r_q2=None, idx=-1):
        d_k = self.args.d_model//self.args.head
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        if a != None:
            len_q = Q.size(2)
            batch_size = Q.size(0)
            n_heads = Q.size(1)
            a_scores = torch.matmul(r_q2, a.transpose(1, 2)).transpose(0, 1)
            a_scores = a_scores.contiguous().view(batch_size, n_heads, len_q, len_q) / np.sqrt(d_k)
            scores += a_scores
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.d_k = self.d_v = args.d_model // args.head
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.head, bias=False)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.head, bias=False)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.head, bias=False)
        self.fc = nn.Linear(args.head * self.d_v, args.d_model, bias=False)
        if args.max_k != -1:
            self.relative_position_k = RelativePosition(self.d_v,
                                                        args.max_k,self.args)  # hid_dim // n_heads,max_relative_position[50,50,1024] [64*2,64*2,1024]?

    def forward(self, Q, K, V, attn_mask, a=None, idx=-1):
        device = 'cuda:%s' % str(self.args.gpu)
        # q: [batch_size, seq_len, d_model] 256,50,1024, k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        len_k = K.shape[1]
        len_q = Q.shape[1]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.args.head, self.d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.args.head, self.d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.args.head, self.d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]
        # print('q q_s shape',Q.shape,q_s.shape)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.head, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        if self.args.max_k == -1:
            a = None
        else:
            a = 1
        if a != None:
            r_k2 = self.relative_position_k(len_k, len_q, self.args.relative_v).to(device)  # .view(batch_size, -1, n_heads, d_k).transpose(1, 2)
            r_q2 = self.W_Q(Q).permute(1, 0, 2).contiguous().view(len_q, batch_size * self.args.head, self.d_k).to(device)
            context = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask, r_k2, r_q2, idx)
        else:
            context = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.args.head * self.d_v)  # context: [batch_size, seq_len, n_heads * d_v]
        output = self.fc(context)
        return nn.LayerNorm(self.args.d_model).to(device)(output + residual)  # output: [batch_size, seq_len, d_model]



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args):
        super(PoswiseFeedForwardNet, self).__init__()
        d_ff = args.d_model*4
        self.fc1 = nn.Linear(args.d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, args.d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)
        self.device = 'cuda:%s' % str(args.gpu)
        self.d_model = args.d_model

    def forward(self, enc_inputs, enc_self_attn_mask, idx=-1):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, enc_inputs,
                                         idx)  # use relative position embedding
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self, args, vocab_size):
        super(BERT, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.embedding = Embedding(args,vocab_size)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.layer)])
       
        self.linear = nn.Linear(args.d_model, args.d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(args.d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight


        if args.use_his:
            self.linear_prior = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.Dropout(0.5))
            self.linear_next = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.Dropout(0.5))
            self.bnorm = nn.BatchNorm1d(args.d_model)

    def forward(self, input_ids, masked_pos, user_ids=None, temp_ids=None, input_prior=None, input_next=None,
                input_next_dis=None, input_prior_dis=None):
        # input_next_dis，input_prior_dis [bach_size,max_pred]
        output = self.embedding(input_ids, user_ids, temp_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, maxlen, maxlen]

        if self.args.use_his:
            input_prior_embedded = self.embedding.tok_embed(input_prior)
            input_next_embedded = self.embedding.tok_embed(input_next)

        for idx, layer in enumerate(self.layers):
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask, idx)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.args.d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        if self.args.use_his:
            if self.args.use_his == 2 or self.args.use_his == 4:
                linear_prior = (self.linear_prior(input_prior_embedded).permute(2, 0, 1) * input_prior_dis).permute(
                    (1, 2, 0))
                linear_next = (self.linear_next(input_next_embedded).permute(2, 0, 1) * input_next_dis).permute(
                    (1, 2, 0))
            elif self.args.use_his == 1 or self.args.use_his == 3:
                linear_prior = (self.linear_prior(input_prior_embedded))
                linear_next = (self.linear_next(input_next_embedded))
            h_masked = self.linear(h_masked) + linear_prior + linear_next  # [batch_size, max_pred, d_model]
            h_masked = self.bnorm(h_masked.permute(0, 2, 1)).permute(0, 2, 1)

        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)
       
        if self.args.is_training not in [2,3,4]:
              # [batch_size, max_pred, vocab_size]
            return logits_lm
        elif self.args.is_training == 2 or self.args.is_training == 4 or self.args.is_training == 3:
            return logits_lm , h_masked  #batch_size, max_pred, vocab_size ,batch size(256), max pred (5), d_model(512)
         
