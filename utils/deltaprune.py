import copy
import torch
import numpy as np
import tqdm
import warnings

def cal_delta_param_decoder(finetune,base):
    tuned = copy.deepcopy(finetune)
    for key in finetune.keys():
        if "mlp" in key or "self_attn" in key:
            base[key].to(tuned[key].dtype)
            if tuned[key].shape != base[key].shape:
                warnings.warn(f"{key} {tuned[key].shape} , {base[key].shape}")
                tuned[key] = tuned[key][:base[key].shape[0]]
                # tuned[key] = tuned[key][:32000]
                tuned[key][:base[key].shape[0]] -= base[key]
            else:
                tuned[key] -= base[key]
    return tuned

def drop_decoder(delta_param_or,mask_rate):
    new_dict = {}
    for key in delta_param_or.keys():
        val_diff = copy.deepcopy(delta_param_or[key])
        if 'lm_head' in key:
            new_dict[key] = copy.deepcopy(delta_param_or[key])
            continue
        ### others to drop 
        mask = torch.bernoulli(torch.full_like(input=val_diff, fill_value=mask_rate)).to(val_diff.device)
        mask = (1 - mask)
        masked_input_tensor = val_diff * mask
        new_dict[key] = masked_input_tensor
    return new_dict

# def add_extra_back(fine_model,new_dict):
#     tuned = copy.deepcopy(fine_model.state_dict())
#     for key in tqdm(tuned):
#         if tuned[key].shape != new_dict[key].shape:
#             warnings.warn(f"{key} {tuned[key].shape} , {new_dict[key].shape}")
#             tuned[key][:new_dict[key].shape[0]] = new_dict[key]
#             new_dict[key] = tuned[key]
#     return new_dict

def add_back_decoder(pretrain,fine_model,delta_dict):
    new_dict = {}
    pre_dict = copy.deepcopy(pretrain.state_dict())
    tuned = copy.deepcopy(fine_model.state_dict())
    for key in delta_dict:
        if key in pre_dict.keys() and ("mlp" in key or "self_attn" in key):
#         print(key)
            masked_input_tensor = delta_dict[key]
            new_dict[key] =  pretrain.state_dict()[key] +  masked_input_tensor.cpu()
            #### add extra back
            if tuned[key].shape != new_dict[key].shape:
                warnings.warn(f"{key} {tuned[key].shape} , {new_dict[key].shape}")
                tuned[key][:new_dict[key].shape[0]] = new_dict[key]
                new_dict[key] = tuned[key]
        else:
            new_dict[key] = tuned[key]
    return new_dict


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

def rescale_v(pretrained_model,drop_dict,q,deocder=False):
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

def rescale_v_decoder(pretrained_model,fintuned_model,drop_dict,q):
    new_dict = {}
    for key in drop_dict.keys():
        if 'lm_head' in key:
            new_dict[key] = copy.deepcopy(drop_dict[key])
            continue
        masked_input_tensor = copy.deepcopy(drop_dict[key])
        masked_input_tensor = torch.div(input=masked_input_tensor, other=q)
        new_dict[key] = masked_input_tensor
    final_dict = add_back_decoder(pretrained_model,fintuned_model,new_dict)
    return final_dict


def rescale_q_decoder(pretrained_model,fintuned_model,drop_dict,q_dict):
    new_dict = {}
    for key in drop_dict.keys():
        if 'classifier' in key or 'lm_head' in key:
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
                top_rate =torch.mean(torch.stack(list(q_dict.values())))
        else:
            top_rate = torch.mean(torch.stack(list(q_dict.values())))
        masked_input_tensor = torch.div(input=masked_input_tensor, other=top_rate)
        new_dict[key] = masked_input_tensor
    final_dict = add_back_decoder(pretrained_model,fintuned_model,new_dict)
    return final_dict

def rescale_q(pretrained_model,drop_dict,q_dict):
    new_dict = {}
    for key in drop_dict.keys():
        if 'classifier' in key or 'lm_head' in key:
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