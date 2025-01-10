# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
import sys
sys.path.append(r'/home/cmkstien/Graphormer/graphormer/modules')
from graphormer_layers import GraphNodeFeature
from fairseq.utils import safe_hasattr


from ..modules import init_graphormer_params, GraphormerGraphEncoder
from ..pretrain import load_pretrained_model
import pickle
logger = logging.getLogger()
logger.setLevel(20)


@register_model("graphormer")
class GraphormerModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)
        self.encoder_embed_dim = args.encoder_embed_dim
        if args.pretrained_model_name != "none":
            self.load_state_dict(load_pretrained_model(args.pretrained_model_name))


        self.freeze_level = args.freeze_level

        # self.freeze_level = 2 ## -x freezes the first x layer

        c = self.freeze_level
        i = 0
        self.freeze_feature_encoder = args.freeze_feature_encoder
        print(self.freeze_feature_encoder)
        if self.freeze_feature_encoder:
            for child in self.encoder.graph_encoder.graph_node_feature.float_encoder.children(): # Freezing the graph feature encoder
                for param in child.parameters():
                    param.requires_grad = False


        if self.freeze_level == 0: ## Do nothing
            x = ':)'
        elif self.freeze_level < 0 : ## Freeze encoder layers
            for child in self.encoder.graph_encoder.layers.children():
                for param in child.parameters():
                    param.requires_grad = False
                c+=1
                if c == 0:
                    break
        elif self.freeze_level > 0: ## Freeze MLP layers: TODO: Implement for encoder head as well (third MLP layer)
            for child in self.encoder.layer_list.children():
                for param in child.parameters():
                    param.requires_grad = False
                c-=1
                if c == 0:
                    break
        # c2 = 0
        # for child in self.encoder.graph_encoder.layers.children():
        #     for param in child.parameters():
        #         print('GRAPHORMER LAYER', c2)
        #         print(param.requires_grad)
        #     c2+=1
        # c3 = 0
        # for child in self.encoder.layer_list.children():
        #     for param in child.parameters():
        #             print('MLP LAYER', c3)
        #             print(param.requires_grad)
        #     c3+=1

        if not args.load_pretrained_model_output_layer:
            self.encoder.reset_output_layer_parameters()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--freeze-level", type=int, metavar="D", help="LAYERS TO FREEZE. 0 does nothing, negative values freeze encoder layers starting from the front, and positive values freeze the MLP"
        )

        parser.add_argument(
            "--freeze-feature-encoder", type=bool, default=False, metavar="D", help="Freeze the feature encoder"
        )

        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input" " and output embeddings",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn",
        
        )
        
        parser.add_argument(
            "--mlp-layers",
            type=int,
            help="number of layers in the mlp",

        )

        # Arguments related to parameter initialization
        parser.add_argument(
            "--apply-graphormer-init",
            action="store_true",
            help="use custom param initialization for Graphormer",

        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--pre-layernorm",
            action="store_true",
            help="apply layernorm before self-attention and ffn. Without this, post layernorm will used",
        )

    def max_nodes(self):
        return self.encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = GraphormerEncoder(args)
        return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_nodes = args.max_nodes
        self.edge_encodings = nn.Embedding(16 + 1, 128, None)
        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
        )

        self.layers = args.mlp_layers
        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None
        self.layer_list = torch.nn.ModuleList()
        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)
        latent_size = 2100
        w = 10**20
        for i in range(self.layers - 1):
                ln = nn.Linear(
                latent_size,latent_size)
                self.layer_list.append(ln)

        kernel = 40
        self.embed_out =nn.Linear(
                    latent_size, args.num_classes,bias=True)#+ kernel - 1, )
        w = 10^10

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel)
        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))


    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()


    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):

        c2 = 0

        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )


        x = inner_states[-1].transpose(0, 1)[:,0,:] ## grabs graph token 

        if masked_tokens is not None:
            raise NotImplementedError
        
        for i, layer in enumerate(self.layer_list): #
            x = layer(x)
            x = F.relu(x) 

        x = self.embed_out(x)
        x = torch.unsqueeze(x, 1)

        return x

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict

@register_model_architecture("graphormer", "graphormer")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)


@register_model_architecture("graphormer", "graphormer_base")
def graphormer_base_architecture(args):
    if args.pretrained_model_name == "pcqm4mv1_graphormer_base" or \
       args.pretrained_model_name == "pcqm4mv2_graphormer_base" or \
       args.pretrained_model_name == "pcqm4mv1_graphormer_base_for_molhiv":
        args.encoder_layers = 12
        args.encoder_attention_heads = 32
        args.encoder_ffn_embed_dim = 768
        args.encoder_embed_dim = 768
        args.dropout = getattr(args, "dropout", 0.0)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.act_dropout = getattr(args, "act_dropout", 0.1)
    else:
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
        args.encoder_layers = getattr(args, "encoder_layers", 12)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
        args.dropout = getattr(args, "dropout", 0.0)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.act_dropout = getattr(args, "act_dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)


@register_model_architecture("graphormer", "graphormer_slim")
def graphormer_slim_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 80)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 80)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)


@register_model_architecture("graphormer", "graphormer_large")
def graphormer_large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)

    args.encoder_layers = getattr(args, "encoder_layers", 24)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)
