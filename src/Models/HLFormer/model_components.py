import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None, contexts=None, queries=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        if self.reduction:
            return torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)
        else:
            return t2v_denominator - t2v_nominator


class frame_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(frame_nce, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):

        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)

        nominator = torch.logsumexp(nominator, dim=1)

        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class GaussianBlock(nn.Module):
    def __init__(self, config):
        super(GaussianBlock, self).__init__()
        
        self.num_block = config.attention_num - 1
        self.e_attns = nn.ModuleList()
        self.e_attns.append(EuclideanAttentionBlock(config))
        for i in range(1, self.num_block + 1):
            wid = 2 ** (i)
            self.e_attns.append(EuclideanAttentionBlock(config, wid=wid))

        self.ca = CrossAttention(config)
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer2 = nn.Linear(config.hidden_size, 1)

        self.sft_factor = config.sft_factor
        self.branch_norm = nn.LayerNorm(config.hidden_size)
        self.weight_token_mode = getattr(config, "weight_token_mode", "global")
        _hybrid_init = float(getattr(config, "weight_token_hybrid_init", 0.7))
        _hybrid_init = min(max(_hybrid_init, 1e-4), 1 - 1e-4)
        self.weight_token_hybrid_logit = nn.Parameter(
            torch.tensor(math.log(_hybrid_init / (1.0 - _hybrid_init)), dtype=torch.float32)
        )

    def forward(self, input_tensor, attention_mask=None, weight_token=None):

        outputs = []
        for i in range(len(self.e_attns)):
            o = self.branch_norm(self.e_attns[i](input_tensor, attention_mask)).unsqueeze(-1)
            outputs.append(o)
        oo = torch.cat(outputs, dim=-1)

        B, L, H, K = oo.shape
        if attention_mask is not None:
            valid = attention_mask.squeeze(1).type_as(oo)
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_token = (oo.mean(dim=-1) * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        else:
            mean_token = oo.mean(dim=-1).mean(dim=1, keepdim=True)

        if weight_token is not None:
            global_token = weight_token.to(oo.device).type_as(oo)
            if global_token.shape[0] != B:
                global_token = global_token.expand(B, -1, -1)
        else:
            global_token = mean_token

        if self.weight_token_mode == "mean":
            selected_token = mean_token
        elif self.weight_token_mode == "hybrid":
            gate = torch.sigmoid(self.weight_token_hybrid_logit)
            selected_token = gate * global_token + (1.0 - gate) * mean_token
        else:
            selected_token = global_token

        branch_feat = oo.permute(0, 3, 1, 2).reshape(B * K, L, H)
        query_token = selected_token.repeat_interleave(K, dim=0)
        branch_mask = attention_mask.repeat_interleave(K, dim=0) if attention_mask is not None else None
        weight = self.ca(query_token, branch_feat, branch_mask).view(B, K, H)

        weight = self.layer1(weight)
        weight = self.dropout(F.relu(weight))
        weight = self.layer2(weight).squeeze(-1)

        weight = F.softmax(weight / self.sft_factor, dim=-1)
        out = torch.sum(oo * weight[:, None, None, :], dim=-1)

        return out


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x

class EuclideanAttentionBlock(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanAttentionBlock, self).__init__()
        self.self = EuclideanGaussianAttention(config, wid=wid)
        self.output = FeedForward(config.hidden_size, int(4*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        self_output = self.dropout1(self_output)
        input_tensor = self.norm1(input_tensor + self_output)
        tmp = self.output(input_tensor)
        tmp = self.dropout2(tmp)
        input_tensor = self.norm2(input_tensor + tmp)
        return input_tensor


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.self = EuclideanGaussianAttention(config)
        self.output = FeedForward(config.hidden_size, int(1*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, query, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(query, input_tensor, input_tensor, attention_mask)

        self_output = self.dropout1(self_output)
        query = self.norm1(query + self_output)
        tmp = self.output(query)
        tmp = self.dropout2(tmp)
        query = self.norm2(query + tmp)
        return query


class EuclideanGaussianAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanGaussianAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.wid = wid
        self._gauss_cache = {}
        _num_block = config.attention_num - 1
        self.gauss_divisor = 2 ** (_num_block) + 1
        self.bias_mode = getattr(config, "gauss_bias_mode", "add_log")
        self._log_gauss_cache = {}

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def generate_gauss_weight(self, props_len, width, device, dtype):
        cache_key = (props_len, float(width), str(device), str(dtype))
        if cache_key in self._gauss_cache:
            return self._gauss_cache[cache_key]

        center = torch.arange(props_len, device=device, dtype=dtype) / props_len
        width_t = (width * torch.ones(props_len, device=device, dtype=dtype)).unsqueeze(-1).clamp(1e-2) / self.gauss_divisor
        weight = torch.linspace(0, 1, props_len, device=device, dtype=dtype)
        weight = weight.view(1, -1).expand(center.size(0), -1)
        center = center.unsqueeze(-1)

        w = 0.3989422804014327
        gauss = w / width_t * torch.exp(-(weight - center) ** 2 / (2 * width_t ** 2))
        gauss = gauss / gauss.max(dim=-1, keepdim=True)[0]
        self._gauss_cache[cache_key] = gauss
        return gauss

    def generate_log_gauss_bias(self, props_len, width, device, dtype):
        cache_key = (props_len, float(width), str(device), str(dtype))
        if cache_key in self._log_gauss_cache:
            return self._log_gauss_cache[cache_key]

        gauss = self.generate_gauss_weight(props_len, width, device, dtype)
        log_bias = torch.log(gauss.clamp_min(1e-6))
        self._log_gauss_cache[cache_key] = log_bias
        return log_bias

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores_ori
        if self.wid is not None:
            if self.bias_mode == "add_log":
                log_bias = self.generate_log_gauss_bias(
                    attention_scores.shape[-1], self.wid,
                    device=attention_scores.device,
                    dtype=attention_scores.dtype,
                ).unsqueeze(0).unsqueeze(0)
                attention_scores = attention_scores_ori + log_bias
            else:
                gmm_mask = self.generate_gauss_weight(
                    attention_scores.shape[-1], self.wid,
                    device=attention_scores.device,
                    dtype=attention_scores.dtype,
                ).unsqueeze(0).unsqueeze(0)
                attention_scores = attention_scores_ori * gmm_mask
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
    
