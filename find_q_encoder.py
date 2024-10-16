import torch
import torch.nn as nn
import os
import sys
import json
import argparse
import time
import logging
from functools import partial
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import matplotlib.pyplot as plt
from utils.glue_data_loader import GLUEDataLoader
from utils.utils import *
from utils.getwx import *
from utils.metrics import *
from utils.customized_trainers import CustomizedTrainer
from utils.load_config import cache_dir
from utils.analyticalq import *
from utils.deltaprune import *
import copy
import numpy as np
import random
import os
from torch.utils.data import DataLoader

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed()

def objective_func(inputs,param,args):
    drop_dict = inputs['drop_dict']
    pretrained_model = inputs['pretrained_model']
    if args.analytical:
        q_dict_e = optimal_q_calculation(inputs['metric_score'],inputs['delta_param'],param,p=1-inputs['p'],gpu=args.device)
        delta_2 = rescale_q(pretrained_model,drop_dict,q_dict_e)
    else:
        delta_2 = rescale_v(pretrained_model,drop_dict,param)
        
    finetuned_model_2 = inputs['finetuned_model_2']
    finetuned_model_2.load_state_dict(delta_2)
    dataset_name = inputs['dataset_name']
    trainer = CustomizedTrainer(
            model=finetuned_model_2,              # model
            args=inputs['training_args'],                 # training arguments
            train_dataset=inputs['train_dataset'],        # training dataset
            eval_dataset=inputs['val_dataset'],          # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
            tokenizer=inputs['tokenizer']                 # tokenizer
        )
    test_metrics = trainer.evaluate()
    test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
    pef = extract_res(test_metrics,dataset_name)
    print('the performance of param {} is {}'.format(param,pef))
    return pef

def find_optimal_param(inputs,l_bound,u_bound,step=0.1,decay=0.1):
    """
    step: step size in find the optimal q
    decay: 0-1, the decay rate for more accurate finding q, you can increase it if you need a faster finding.
    """
    best_score = 0
    best_scores = {}
    stop_id = 0
    for initial in  np.arange(l_bound, u_bound+step, step):
        score = objective_func(inputs,initial,args)
        best_scores[initial] = score
        if score >= best_score:
            best_score = score
            stop_id = 0
        else:
            stop_id +=1
        if stop_id > 3:
            break
        
    sorted_slots = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    l_bound = sorted_slots[0][0] - step*(1-decay)
    u_bound = sorted_slots[0][0] + step

    best_score = 0
    stop_id = 0
    for initial in np.arange(l_bound, u_bound, step*decay):
        score = objective_func(inputs,initial,args)
        best_scores[initial] = score
        if score >= best_score:
            best_score = score
            stop_id = 0
        else:
            stop_id +=1
        if stop_id > 3:
            break

    sorted_slots = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_slots[0][0]

    
    


if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--model', type=str, default='ViT-B_16', help='neural network used in training')
        parser.add_argument('--device', type=str, default='cpu', help='device name')
        parser.add_argument('--architect', type=str, default='bert-base-uncased', help='the architecture to be used')
        parser.add_argument('--p', type=float, default=0.01, help='the pruning rate used')
        parser.add_argument('--l_bound', type=float, default=0.1, help='the 1- pruning rate used for q_v, but analynatical different')
        parser.add_argument('--u_bound', type=float, default=0.3, help='upper bound')
        # parser.add_argument('--model_path', type=str, default='cola', help='dataset name to use')
        parser.add_argument('--dataset_name', type=str, default='cola', help='dataset name to use')
        parser.add_argument('--step_size', type=float, default=0.05, help='the pruning rate used')
        parser.add_argument('--finetuned_model', type=str, default=None, help='the finetuned model to use')
        parser.add_argument('--analytical', action='store_true', help='use analytical resolve q')
        parser.add_argument('--param', type=float, default=None, help='the optimal parameter')
        args = parser.parse_args()
        return args
    args = get_args()

    device = args.device
    architect = args.architect
    p=args.p
    dataset_name = args.dataset_name
    ### you fintuned model here
    model_name = args.finetuned_model

    datasets_test_metrics = []

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=architect, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                                train_split_ratio_for_val=0.1,
                                                                                                max_seq_length=128)
    training_args = TrainingArguments(
                output_dir=model_name,                        # save model directory
                per_device_train_batch_size=16,       # batch size per device during training
                per_device_eval_batch_size=16,        # batch size for evaluation
        report_to='none',
            )
    # assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=training_args.output_dir,
                                                                        num_labels=num_labels,output_hidden_states=False)

    pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=architect, cache_dir=cache_dir,
                                                                        num_labels=num_labels,output_hidden_states=False)
    
    finetuned_model_2 =  AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=training_args.output_dir,
                                                                     num_labels=num_labels,output_hidden_states=False)
    
    finetuned_model_2.to(args.device)
    delta_param = cal_delta_param(finetuned_model,pretrained_model)
    delta_param_or = delta_param
    drop_dict = drop(delta_param_or,1-p)
    inputs = {}
    ### find optimal param
    if args.analytical:
        dataloader,_ = get_loaders("wikitext2",tokenizer= tokenizer,seqlen=128,nsamples = 50)
        metric_score = get_wx(finetuned_model.to(args.device),dataloader,delta_param_or)
    if args.param is None:
        if args.analytical:
            inputs['metric_score'] = metric_score
        inputs['train_dataset'] = train_dataset
        inputs['val_dataset'] = val_dataset
        inputs['training_args'] =training_args
        inputs['dataset_name'] = dataset_name
        inputs['tokenizer'] = tokenizer
        inputs['finetuned_model_2'] = finetuned_model_2
        inputs['drop_dict'] = drop_dict
        inputs['pretrained_model'] = pretrained_model
        inputs['delta_param'] = delta_param
        inputs['p'] = p

        param = find_optimal_param(inputs,args.l_bound,args.u_bound,step=args.step_size)
    else:
        param = args.param

    if args.analytical:
        q_dict_e = optimal_q_calculation(metric_score,delta_param,param,p=1-p,gpu=args.device)
    else:
        #### the validaiton q here
        q = param

    res_all = {}
    res_all[p] = []
    ##### drop for 
    for seed in [0,1,2,3]:
        set_seed(seed)
        drop_dict = drop(delta_param_or,1-p)
        if args.analytical:
            delta_2 = rescale_q(pretrained_model,drop_dict,q_dict_e)
        else:
            delta_2 = rescale_v(pretrained_model,drop_dict,param)

        finetuned_model_2.load_state_dict(delta_2)
        trainer = CustomizedTrainer(
                model=finetuned_model_2,              # model
                args=training_args,                 # training arguments
                train_dataset=train_dataset,        # training dataset
                eval_dataset=test_dataset,          # evaluation dataset
                compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
                tokenizer=tokenizer                 # tokenizer
            )
        test_metrics = trainer.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        res_all[p].append(extract_res(test_metrics,dataset_name))
    print(res_all[p])
    print(param)
    print(np.mean(res_all[p]))