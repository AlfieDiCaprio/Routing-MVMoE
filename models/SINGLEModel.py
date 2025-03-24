import torch
import math
import torch.nn as nn
import revtorch as rv

from torch import einsum
from functools import partial

from einops import rearrange
from einops.layers.torch import Rearrange

from scipy.fftpack import next_fast_len
import torch.nn.functional as F
from labml_helpers.module import Module

import logging
logger = logging.getLogger(__name__)


from timm.models.layers import DropPath, trunc_normal_
__all__ = ['SINGLEModel']


class SINGLEModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']

        self.encoder = Rev_Encoder(**model_params)
        self.decoder = SINGLE_Decoder(**model_params)

        self.encoded_nodes = None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        # shape: (batch, problem)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            node_xy_demand_tw = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
            # shape: (batch, problem, 3)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            node_xy_demand_tw = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
            # shape: (batch, problem, 5)
        else:
            raise NotImplementedError

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_tw)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size))
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
            # shape: (batch, pomo, problem_size+1)

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            # selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size).to(self.device)
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            if self.problem in ["CVRP", "VRPB"]:
                attr = state.load[:, :, None]  # shape: (batch, pomo, 1)
            elif self.problem in ["OVRP", "OVRPB"]:
                attr = torch.cat((state.load[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
            elif self.problem in ["VRPTW", "VRPBTW"]:
                attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
            elif self.problem in ["VRPL", "VRPBL"]:
                attr = torch.cat((state.load[:, :, None], state.length[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
            elif self.problem in ["VRPLTW", "VRPBLTW"]:
                attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None]), dim=2)  # shape: (batch, pomo, 3)
            elif self.problem in ["OVRPL", "OVRPBL"]:
                attr = torch.cat((state.load[:, :, None], state.length[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 3)
            elif self.problem in ["OVRPTW", "OVRPBTW"]:
                attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 3)
            elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
                attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 4)
            else:
                raise NotImplementedError
            probs = self.decoder(encoded_last_node, attr, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)
            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = selected
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class Rev_Encoder(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        head_num = self.model_params['head_num']
        intermediate_dim = self.model_params['ff_hidden_dim']
        qkv_dim = self.model_params['qkv_dim']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            self.embedding_node = nn.Linear(3, embedding_dim)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            self.embedding_node = nn.Linear(5, embedding_dim)
        else:
            raise NotImplementedError

        self.num_hidden_layers = encoder_layer_num
        blocks = []
        for _ in range(encoder_layer_num):

            f_func = MLLABlock(embedding_dim, head_num, qkv_bias=True, drop_path=0.)
            g_func = FeedForward(**model_params)
            blocks.append(rv.ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))


    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_tw.shape: (batch, problem, 3/5) - based on self.problem

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand_tw)
        # shape: (batch, problem, embedding)


        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        out = torch.cat([out, out], dim=-1)
        out = self.sequence(out)
        return torch.stack(out.chunk(2, dim=-1))[-1]


########################################
# DECODER
########################################


class SINGLE_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        if self.problem in ["CVRP", "VRPB"]:
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRP", "OVRPB", "VRPTW", "VRPBTW", "VRPL", "VRPBL"]:
            self.Wq_last = nn.Linear(embedding_dim + 2, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPLTW", "VRPBLTW", "OVRPL", "OVRPBL", "OVRPTW", "OVRPBTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        else:
            raise NotImplementedError
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, attr, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, 1~4)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

# functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    return val if exists(val) else d

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "GatedRMSNorm":
            self.norm = GatedRMSNorm(embedding_dim)
        elif model_params["norm"] == "simpleRMSNorm":
            self.norm = SimpleRMSNorm(embedding_dim)
        elif model_params["norm"] == "RMSNorm":
            self.norm = RMSNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, GatedRMSNorm):
            added = input1 + input2 if self.add else input2
            normalized = self.norm(added)
        elif isinstance(self.norm, SimpleRMSNorm):
            added = input1 + input2 if self.add else input2
            normalized = self.norm(added)
        elif isinstance(self.norm, RMSNorm):
            added = input1 + input2 if self.add else input2
            normalized = self.norm(added)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class MLLABlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, drop_path=0., **kwargs):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads

        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)

        self.act = get_activation_fn("leaky_relu")

        self.NormLinearAttention = NormLinearAttention(embed_dim=dim, hidden_dim=dim*4, num_heads=num_heads)

        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):

        shortcut = x

        act_res = self.act(self.act_proj(x))

        x = self.in_proj(x)

        x = self.act(x)

        x = self.NormLinearAttention(x)

        x = self.out_proj(x * act_res)

        x = shortcut + self.drop_path(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"



########################################
# Norm Linear Attention
# Code: https://github.com/Doraemonzzz/transnormer-v2-pytorch/blob/main/transnormer_v2/norm_linear_attention.py
# Paper: https://www.semanticscholar.org/reader/e3fc46d5f4aae2c7a8a86b6bd21ca8db5d40fcbd
########################################

def get_activation_fn(activation):
    logger.info(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            # 调用自定义函数F.elu对x进行处理
            # F.elu是一个自定义函数，可能用于计算指数线性单元（ELU）激活函数
            return 1 + F.elu(x)
        return f
    elif activation == "2+elu":
        def f(x):
            return 2 + F.elu(x)
        return f
    elif activation == "silu":
        return F.silu
    elif activation == "Leaky ReLU":
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    else:
        return lambda x: x

def get_norm_fn(norm_type):
    if norm_type == "layernorm":
        return nn.LayerNorm
    elif norm_type == "batchnorm":
        return nn.BatchNorm1d
    elif norm_type == "instancenorm":
        return nn.InstanceNorm1d
    else:
        # 默认返回 layernorm 或者不做处理
        return nn.BatchNorm1d



class LaplacianAttnFn(nn.Module):
    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

####################
# Normalization -- RMSNorm / SimpleRMSNorm / GatedRMSNorm
####################


class SimpleRMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(SimpleRMSNorm, self).__init__()
        self.eps = eps
        self.d = d

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        return x_normed



class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class GatedRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(GatedRMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        self.gate = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)


    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed * torch.sigmoid(self.gate * x)



####################
# Attention module -- FocusedLinearAttention
####################



####################
# FFN module -- SwishGLU
####################


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

####################
# Ablation: LinearAttentionfor
####################

class LinearAttention(nn.Module):
    """ Simplified Linear Attention.

    Args:
        dim (int): Number of input channels (embedding dimension).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子，用于稳定训练
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 生成 Q, K, V
        self.out_proj = nn.Linear(dim, dim)  # 输出投影

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, sequence_length, embedding_dim)
        Returns:
            Tensor with shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length, embedding_dim = x.shape
        assert embedding_dim == self.dim, "Input embedding_dim must match initialized dim"

        # Step 1: Compute Q, K, V
        qkv = self.qkv(x)  # Shape: (batch_size, sequence_length, 3 * embedding_dim)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Shapes: (batch_size, num_heads, sequence_length, head_dim)

        # Step 2: Linear Attention
        # Apply softmax along sequence_length for K
        k = k.softmax(dim=-2)  # Normalize along sequence length
        q = q * self.scale  # Scale Q

        # Compute attention scores: (batch_size, num_heads, sequence_length, sequence_length)
        context = torch.einsum("bhlk,bhlv->bhlv", k, v)  # Weighted sum of V
        x = torch.einsum("bhlv,bhlk->bhlv", q, context)  # Q * Context

        # Step 3: Reshape and project output
        x = x.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, embedding_dim)  # Combine heads
        x = self.out_proj(x)  # Final linear projection

        return x

########################################
# Norm Linear Attention (for CTSP)
# Code: https://github.com/Doraemonzzz/transnormer-v2-pytorch/blob/main/transnormer_v2/norm_linear_attention.py
# Paper: https://www.semanticscholar.org/reader/e3fc46d5f4aae2c7a8a86b6bd21ca8db5d40fcbd
########################################

def get_activation_fn(activation):
    logger.info(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            # 调用自定义函数F.elu对x进行处理
            # F.elu是一个自定义函数，可能用于计算指数线性单元（ELU）激活函数
            return 1 + F.elu(x)
        return f
    elif activation == "2+elu":
        def f(x):
            return 2 + F.elu(x)
        return f
    elif activation == "silu":
        return F.silu
    elif activation == "Leaky ReLU":
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    else:
        return lambda x: x

def get_norm_fn(norm_type):
    if norm_type == "layernorm":
        return nn.LayerNorm
    elif norm_type == "batchnorm":
        return nn.BatchNorm1d
    elif norm_type == "instancenorm":
        return nn.InstanceNorm1d
    else:
        # 默认返回 layernorm 或者不做处理
        return nn.BatchNorm1d


class NormLinearAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_heads,
            act_fun="leaky_relu",
            uv_act_fun="silu",
            # act_fun="elu",
            # uv_act_fun="swish",
            # norm_type="instancenorm", # optional: layernorm / batchnorm / instancenorm
            norm_type="layernorm", # optional: layernorm / batchnorm / instancenorm
            causal=False,
    ):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        self.u_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = get_activation_fn(act_fun)
        self.uv_act = get_activation_fn(uv_act_fun)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.focusing_factor = 3

        self.scale = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))

        # self.norm = get_norm_fn(norm_type)(hidden_dim)
        self.norm_type = norm_type
        NormLayer = get_norm_fn(norm_type)
        if norm_type == "layernorm":
            # LayerNorm 通常直接传维度 hidden_dim 即可
            self.norm = NormLayer(hidden_dim)
        else:
            # BatchNorm1d / InstanceNorm1d 对“通道数”做归一化，这里相当于 hidden_dim 为 channel
            self.norm = NormLayer(hidden_dim, affine=True)
        self.causal = causal

    def forward(
            self,
            x,
            y=None,
            attn_mask=None,
    ):
        # x: b n d
        if y == None:
            y = x
        n = x.shape[-2]
        # linear map
        q = self.q_proj(x)
        u = self.u_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        # uv act
        u = self.uv_act(u)
        v = self.uv_act(v)
        # reshape
        q, k, v = map(lambda x: rearrange(x, '... n (h d) -> ... h n d', h=self.num_heads), [q, k, v])
        # act


        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()

        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6

        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale

        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)


        q = q ** focusing_factor
        k = k ** focusing_factor

        q = (q / q.norm(dim=-1, keepdim=True)) * q.norm(dim=-1, keepdim=True)
        k = (k / k.norm(dim=-1, keepdim=True)) * k.norm(dim=-1, keepdim=True)

        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.tril(torch.ones(n, n))).to(q)
            l1 = len(q.shape)
            l2 = len(attn_mask.shape)
            for _ in range(l1 - l2):
                attn_mask = attn_mask.unsqueeze(0)
            energy = torch.einsum('... n d, ... m d -> ... n m', q, k)
            energy = energy * attn_mask
            output = torch.einsum('... n m, ... m d -> ... n d', energy, v)
        else:
            kv = torch.einsum('... n d, ... n e -> ... d e', k, v)
            output = torch.einsum('... n d, ... d e -> ... n e', q, kv)

        output = rearrange(output, '... h n d -> ... n (h d)')

        if self.norm_type == "layernorm":
            output = self.norm(output)
        else:

            output = output.transpose(1, 2)  # -> (B, hidden_dim, N)
            output = self.norm(output)
            output = output.transpose(1, 2)  # -> (B, N, hidden_dim)

        output = u * output

        output = self.out_proj(output)

        return output