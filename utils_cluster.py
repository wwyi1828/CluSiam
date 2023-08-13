import torch
import torch.nn.functional as F
import numpy as np



def final_cluster_similarity(representations, assigns, num_clusters=100, eps=1e-8):

    representations = F.normalize(representations, dim=1, eps=eps)
    soft_assigns = F.gumbel_softmax(assigns, hard=False)
    #soft_assigns = F.softmax(assigns)

    max_index = soft_assigns.max(dim=-1, keepdim=True)[1]
    hard_assigns = torch.zeros_like(soft_assigns, memory_format=torch.legacy_contiguous_format).scatter_(-1, max_index, 1.0)
    hard_assigns = hard_assigns - soft_assigns.detach() + soft_assigns

    center_feats = torch.matmul(hard_assigns.T,representations)
    center_feats /= torch.sum(hard_assigns,dim=0).unsqueeze(1).maximum(torch.tensor(1))
    center_feats = F.normalize(center_feats, dim=1, eps=eps)

    outer_simi = torch.matmul(center_feats,center_feats.T)
    outer_simi = torch.triu(outer_simi,diagonal=1) # 1: zero diagonal elements
    avg_outer = outer_simi.flatten()[torch.nonzero(outer_simi.flatten())].mean()
    avg_outer = torch.nan_to_num(avg_outer,1)

    return avg_outer

