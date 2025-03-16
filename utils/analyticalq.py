import numpy as np
import torch
import copy

def phi_func(x):
    return (1-2*x)/np.log((1-x)/x)

def analytical_resolve_new(p,c,eta= 0.45,prob=0.95,stop_iter= 0.8):
    """
    p: pruning rate
    c: weight x activations
    eta: paramter to balance two in equalty
    prob: gamma probability
    stop_iter: avoid the influence of outlier features,stop when stop_iter % dimension got the best q. adjust it if your model has more outliers
    """
    q_o = torch.ones(c.shape[0], 1)*(1-p)
    stop_index = torch.zeros(c.shape[0], 1)
    best_res = torch.ones_like(q_o,dtype=torch.float16)* 10000
    for q_100 in range(int((1-p)*1000),999):
        q = q_100*0.001
        #### object function for optimal q under an eta
        term1 = (1-(1-p)/q)*torch.sum(c,dim=1,keepdim = True)
        term2 = eta**2*phi_func(p)*torch.sum(c**2,dim=1,keepdim = True)/(4*q**2) + np.log(2/(1-prob))
        res = eta*term1 + term2
        res = abs(res).cpu()
        index_s = res <= best_res
        index_l = res >= best_res
        best_res[index_s] = copy.deepcopy(res[index_s])
        q_o[index_s] = q
        stop_index[index_l] =1
        #### avoid the influence of outlier features 
        if stop_index.mean() >= stop_iter:
            break
    return q_o


def optimal_q_calculation(metric_score,delta_param,eta,p=0.99,gpu='cuda:2',stop_iter = 0.8):
    q_dict_e = {}
    q_dict_or = {}
    for name in delta_param.keys():
        ### skip parameters that don't take inputs and paraneters won't been pruned
        if 'lm_head' in name:
            continue
        if 'classifier' in name:
            continue
        if 'bias'  in name:
            continue
        if 'LayerNorm'  in name:
            continue
        if 'embeddings' in name:
            continue
        if name in metric_score.keys():
            c = (metric_score[name]).to(torch.float16)
            name_b = copy.deepcopy(name)
            name_b = name_b.replace("weight", "bias")
            if name_b in delta_param.keys():
                c_b = delta_param[name_b].unsqueeze(1).to(torch.float16)
                c = torch.cat([c,c_b],dim=1)
            c = c.to(gpu)
            best_2 = analytical_resolve_new(p,c,eta=eta,stop_iter=stop_iter)
            q_dict_or[name] = best_2
            q_dict_e[name] = best_2.mean()
    return q_dict_e
