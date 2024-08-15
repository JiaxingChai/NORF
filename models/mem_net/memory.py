import math
import torch
from typing import Optional


def get_similarity(key, key_scale, query, query_select):
    # used for training/inference and memory reading/memory potentiation
    # key:              B x CK x [N]    - reference feature keys
    # scale_term:       B x  1 x [N]    - reference feature scale term
    # query:            B x CK x [HW/P] - 
    # select_term:      B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    C = key.shape[1]
    key = key.flatten(start_dim=2)
    key_scale = key_scale.flatten(start_dim=1).unsqueeze(2) if key_scale is not None else None
    query = query.flatten(start_dim=2)
    query_select = query_select.flatten(start_dim=2) if query_select is not None else None

    if query_select is not None:
        # See appendix for derivation
        # or you can just trust me ヽ(ー_ー )ノ
        key = key.transpose(1, 2)
        a_sq = (key.pow(2) @ query_select)
        two_ab = 2 * (key @ (query * query_select))
        b_sq = (query_select * query.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq+two_ab-b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = key.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (key.transpose(1, 2) @ query)
        similarity = (-a_sq+two_ab)

    if key_scale is not None:
        similarity = similarity * key_scale / math.sqrt(C)   # B*N*HW
    else:
        similarity = similarity / math.sqrt(C)   # B*N*HW

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(key, key_scale, query, query_select):
    similarity = get_similarity(key, key_scale, query, query_select)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, value):
    B, C, H, W = value.shape

    mo = value.view(B, C, H*W) 
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, C, H, W)

    return mem


if __name__ == "__main__":
    key = torch.randn(2, 32, 64, 64).to('cuda:1')
    key_scale = torch.randn(2, 1, 64, 64).to('cuda:1')
    query = torch.randn(2, 32, 64, 64).to('cuda:1')
    query_select = torch.randn(2, 32, 64, 64).to('cuda:1')
    value = torch.randn(2, 32, 1, 64, 64).to('cuda:1')
    
    affinity = get_affinity(key, key_scale, query, query_select)
    print(affinity.shape)
    memory = readout(affinity, value)
    print(memory.shape)
    
    