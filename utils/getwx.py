# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
import torch
import torch.nn as nn
import copy

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append(inp)
    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp,output):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row +=inp.mean(1)  / self.nsamples
#         self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_wx(finetuned_model,inputs,delta_param_or):
    nsamples = len(inputs)
    metric_score = {}
    subset = find_layers(finetuned_model)
    wrapped_layers = {}
    for name in subset:
        wrapped_layers[name] = WrappedGPT(subset[name])
    def add_batch(name):
        def tmp(_,inp,output):
            wrapped_layers[name].add_batch(inp[0].data,None)
        return tmp
    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    for j in range(nsamples):
        with torch.no_grad():
            _ = finetuned_model(inputs[j].to(finetuned_model.device))
    for name in subset:
        W_metric = delta_param_or[name+'.weight'] * wrapped_layers[name].scaler_row.reshape((1,-1)).cpu()
        metric_score[name+'.weight'] = copy.deepcopy(W_metric)

    return metric_score

def prepare_calibration_input(model, dataloader,seqlen,num_sample, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((num_sample, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def get_wx_decoder(f_model,inputs,tuned_delta,nsamples,seqlen):
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(f_model, inputs,seqlen,nsamples, f_model.device)
    layers = f_model.model.layers
    layer_rap = {}
    metric_score = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        # inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                _ = layer(inps[j].unsqueeze(0).to(f_model.device), attention_mask=attention_mask, position_ids=position_ids)
                outs[j] = layer(inps[j].unsqueeze(0).to(f_model.device), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        for name in subset:
            # print(f"pruning layer {i} name {name}")
            # f_W_metric = f_model.state_dict()['model.layers.'+str(i)+ '.'+name+'.weight'].detach().cpu() * wrapped_layers[name].scaler_row.reshape((1,-1)).detach().cpu()
            W_metric = tuned_delta['model.layers.'+str(i)+ '.'+name+'.weight'] * wrapped_layers[name].scaler_row.reshape((1,-1)).detach().cpu()
            metric_score['model.layers.'+str(i)+ '.'+name+'.weight'] = copy.deepcopy(W_metric)
        inps, outs = outs, inps
        layer_rap[i] = copy.deepcopy(wrapped_layers)
        
    return metric_score