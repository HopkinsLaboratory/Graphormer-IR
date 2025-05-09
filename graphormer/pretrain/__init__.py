from torch.hub import load_state_dict_from_url
from torch.hub import load
import torch.distributed as dist
from torch import load

PRETRAINED_MODEL_URLS = {
    "pcqm4mv1_graphormer_base":"https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/checkpoint_best_pcqm4mv1.pt",
    "pcqm4mv2_graphormer_base":"https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv2/checkpoint_best_pcqm4mv2.pt",
    "oc20is2re_graphormer3d_base":"https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/checkpoint_last_oc20_is2re.pt",
    "pcqm4mv1_graphormer_base_for_molhiv":"https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}

def load_pretrained_model(pretrained_model_name):
    pretrained_model = load(pretrained_model_name)
    # pretrained_model = load('/home/cmkstien/Graphormer_RT_extra/best_MODEL_ext/seeds/external_out_final_cbest/' + pretrained_model_name)
    print(pretrained_model['model'].keys())
    print(pretrained_model['model']['encoder.graph_encoder.graph_node_feature.float_encoder.0.0.weight'])
    # if pretrained_model_name not in PRETRAINED_MODEL_URLS:
    #     raise ValueError("Unknown pretrained model name %s", pretrained_model_name)
    # if not dist.is_initialized():
    #     return load_state_dict_from_url(PRETRAINED_MODEL_URLS[pretrained_model_name], progress=True)["model"]
    # else:
    #     pretrained_model = load_state_dict_from_url(PRETRAINED_MODEL_URLS[pretrained_model_name], progress=True, file_name=f"{pretrained_model_name}_{dist.get_rank()}")["model"]
    #     dist.barrier()
    # keys = ["criterion"]#, , "optimizer_history", "task_state", "extra_state", "last_optimizer_state"] ## "model"
    # delete = ["args","cfg"]
    # for i in keys:
    #     print(pretrained_model[i])
    #     # if i in pretrained_model:
    #     #     del pretrained_model[i]
    # for i in delete:
    #     if i in pretrained_model:
    #         del pretrained_model[i]
    # exit()

    return pretrained_model["model"]