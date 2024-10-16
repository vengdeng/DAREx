import copy
import torch
import numpy as np


def cal_delta_param(model_f,model):
    new_dict = {}
    for key in model_f.state_dict():
        if 'classifier' in key:
            val_f = model_f.state_dict()[key]
            new_dict[key] = copy.deepcopy(val_f)
        else:
            val_f = model_f.state_dict()[key]
            val_p = model.state_dict()[key]
            val_diff = val_f - val_p
            new_dict[key] = copy.deepcopy(val_diff)
    return new_dict

def add_back(pretrain,delta_dict):
    new_dict = {}
    pre_dict = copy.deepcopy(pretrain.state_dict())
    for key in delta_dict:
        if key in pre_dict.keys() and 'classifier' not in key:
#         print(key)
            masked_input_tensor = delta_dict[key]
            new_dict[key] =  pretrain.state_dict()[key] +  masked_input_tensor.cpu()
        else:
            new_dict[key] = delta_dict[key]
    return new_dict

def drop(delta_param_or,mask_rate):
    new_dict = {}
    for key in delta_param_or.keys():
        val_diff = copy.deepcopy(delta_param_or[key])
        if 'classifier' in key:
            new_dict[key] = copy.deepcopy(delta_param_or[key])
            continue
        ### others to drop 
        mask = torch.bernoulli(torch.full_like(input=val_diff, fill_value=mask_rate)).to(val_diff.device)
        mask = (1 - mask)
        masked_input_tensor = val_diff * mask
        new_dict[key] = masked_input_tensor
    return new_dict

def rescale_v(pretrained_model,drop_dict,q):
    new_dict = {}
    for key in drop_dict.keys():
        if 'classifier' in key:
            new_dict[key] = copy.deepcopy(drop_dict[key])
            continue
        masked_input_tensor = copy.deepcopy(drop_dict[key])
        masked_input_tensor = torch.div(input=masked_input_tensor, other=q)
        new_dict[key] = masked_input_tensor
    final_dict = add_back(pretrained_model,new_dict)
    return final_dict

def rescale_q(pretrained_model,drop_dict,q_dict):
    new_dict = {}
    for key in drop_dict.keys():
        if 'classifier' in key:
            new_dict[key] = copy.deepcopy(drop_dict[key])
            continue
        masked_input_tensor = copy.deepcopy(drop_dict[key])
        if key in q_dict.keys() :
            top_rate = q_dict[key]
        elif 'bias' in key:
            name_b = copy.deepcopy(key)
            name_b = name_b.replace("bias", "weight")
            if name_b in q_dict.keys():
                top_rate = q_dict[name_b]
            else:
                top_rate =np.mean(list(q_dict.values()))
        else:
            top_rate = np.mean(list(q_dict.values()))
        masked_input_tensor = torch.div(input=masked_input_tensor, other=top_rate)
        new_dict[key] = masked_input_tensor
    final_dict = add_back(pretrained_model,new_dict)
    return final_dict