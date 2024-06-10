# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )


        self.d_relation = self.num_heads
        self.norm_act = nn.LayerNorm(embed_dim)
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.linear_q = quant_noise(nn.Linear(embed_dim, self.d_relation,bias=bias), q_noise, qn_block_size)
        self.linear_k = quant_noise(nn.Linear(embed_dim, self.d_relation,bias=bias), q_noise, qn_block_size)
        self.linear_v = quant_noise(nn.Linear(embed_dim, embed_dim,bias=bias), q_noise, qn_block_size)

        self.talking = nn.Linear(self.d_relation, self.d_relation, bias=False)

        self.output_layer = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.linear_q.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.linear_k.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.linear_v.weight, gain=1 / math.sqrt(2))


        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.linear_q.weight)
            nn.init.xavier_uniform_(self.linear_k.weight)
            nn.init.xavier_uniform_(self.linear_v.weight)



        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # if key is not None:
        #     src_len, key_bsz, _ = key.size()
        #     if not torch.jit.is_scripting():
        #         assert key_bsz == bsz
        #         assert value is not None
        #         assert src_len, bsz == value.shape[:2]
        # print(query.shape, "QUEEEEERY")
        query = query.transpose(0,1)
        # print(query.shape, "INITIAL QUERY")
        node_q = self.linear_q(query).contiguous().view(bsz, -1, self.d_relation, 1)                                 # [b, len, d_srel, 1] ## ADD CONTINIGUOUS TO THIS
        node_k = self.linear_k(query).contiguous().view(bsz, -1, 1, self.d_relation)                                # [b, len, 1, d_srel]
        node_v = self.linear_v(query).contiguous().view(bsz, -1, self.d_relation, self.embed_dim//self.d_relation)
        # print(q.shape, v.shape, k.shape, "STARTING SHAPES QKV")
        node_q *= self.scaling
        # print(node_q.shape, node_k.shape, node_v.shape, "QKV PRe TRANSFORM")
        node_q = node_q.transpose(1, 2)   # [b, d_srel, len, 1]
        node_k = node_k.transpose(1, 3)   # [b, d_srel, 1, len]
        node_v = node_v.transpose(1, 2)   # [b, h, len, d_v]
        # print(node_q.shape, node_k.shape, node_v.shape, "QKV AFTER TRANSFORM")
        relation = torch.matmul(node_q, node_k)
                                                         # [b, d_srel, q_len, k_len]


        # print(relation.shape, "J")
        relation = relation.permute(0,2,3,1) 
        relation = self.talking(relation)

        if attn_bias is not None:
            attn_bias = attn_bias.permute(0,2,3,1) 
            # print(relation.shape, attn_bias.shape)
            # exit()
            relation += attn_bias

        if key_padding_mask is not None:
            # print(relation.shape)
            # relation = relation.view(bsz, self.num_heads, tgt_len, src_len)
            # print(relation.shape)
            mask = attn_bias.isinf()
            # print(mask.shape)
            # print(relation.shape)

            relation= relation.masked_fill(
                mask.to(torch.bool),
                float("-inf"),
            )
        # print(relation.shape, "POST MASK")
        ## TO DO: 
        ## Ask Zeping to help debug?
        relation = relation.softmax(dim=2)                          # [b, q_len, k_len, d_srel]
        # print(relation)
        # exit()
        relation = relation.permute(0,3,1,2)  
        relation = self.dropout_module(relation)

        node = torch.matmul(relation, node_v)
        node = node.transpose(1,2).reshape(bsz, -1, self.embed_dim)
        node = self.output_layer(node)
        # print(node.shape, "AFTER FINAL PROJECT")
        node = node.transpose(0, 1)
        # print(node.shape, "FINAL")
        # exit()

        # print(node)
        
        attn_weights: Optional[Tensor] = None
        # print(node)
        # exit()
        return node, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value