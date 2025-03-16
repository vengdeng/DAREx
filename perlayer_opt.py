import argparse
import copy
import json
import jsonlines
import os
import random
import re
import sys
from fraction import Fraction

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)
from vllm import LLM, SamplingParams

from utils.deltaprune import *
from utils.resserve import *
from utils.utils import EarlyStopping,top_k_mse_loss
from find_q_decoder import batch_generation,cal_acc_gsm8k,get_data
# Maximum integer value
MAX_INT = sys.maxsize

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune pruned model with delta parameters."
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Pretrained model path or identifier."
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="qwen05b_model_baseline_nospeed/checkpoint-5000/",
        help="Path to the finetuned model checkpoint."
    )
    parser.add_argument(
        "--device_f_model",
        type=str,
        default="cuda:0",
        help="Device for the original finetuned model."
    )
    parser.add_argument(
        "--device_f_model2",
        type=str,
        default="cuda:6",
        help="Device for the pruned model."
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.99,
        help="Probability threshold for drop decoder."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--scheduler_T_max",
        type=int,
        default=200,
        help="T_max parameter for CosineAnnealingLR scheduler."
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Patience for early stopping."
    )
    parser.add_argument(
        "--early_stopping_save_path",
        type=str,
        default="best_model.pth",
        help="File path to save the best model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai/gsm8k",
        help="Dataset name to load from the Hugging Face hub."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Which split of the dataset to use."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the pretrained and finetuned models along with tokenizer.
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model,
        output_hidden_states=False,
        device_map="cpu"
    )
    f_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.finetuned_model,
        output_hidden_states=False,
        device_map="cpu"
    )
    f_model2 = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.finetuned_model,
        output_hidden_states=False,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.finetuned_model,
        padding_side='left'
    )

    # Calculate delta parameters and apply pruning
    tuned_delta = cal_delta_param_decoder(f_model.state_dict(), model.state_dict())
    drop_dict = drop_decoder(tuned_delta, args.p)
    res_diff(model, f_model2, drop_dict, 1 - args.p)

    # Load and preprocess the dataset
    ds_train = load_dataset(args.dataset, "main", split=args.dataset_split)
    ds_train = ds_train.train_test_split(test_size=0.05, shuffle=True, seed=42)
    ds_train, ds_val = ds_train["train"], ds_train["test"]

    batch_size = args.batch_size
    remove_columns = ['question', 'answer']
    # Note: `preprocess` should be defined elsewhere or imported.
    dataset = ds_train.map(
        lambda examples: preprocess(tokenizer, examples),
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=default_data_collator,
    )
    dataset_val = ds_val.map(
        lambda examples: preprocess(tokenizer, examples),
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size*2,
        num_workers=0,
        collate_fn=default_data_collator,
    )

    # Freeze all parameters in f_model2 except those with 'coeff' in their name.
    for k, v in f_model2.named_parameters():
        if 'coeff' not in k:
            v.requires_grad = False
    update_params = [v for k, v in f_model2.named_parameters() if v.requires_grad]

    # Move models to the specified devices.
    f_model.to(args.device_f_model)
    f_model2.to(args.device_f_model2)

    optimizer = torch.optim.AdamW(update_params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_T_max)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        save_path=args.early_stopping_save_path
    )

    train_loss_list = []
    bar = tqdm.tqdm(dataloader, desc="Training")


    # Training loop
    for step, batch in enumerate(bar):
        # Process batch for f_model
        batch1 = {k: v.to(f_model.device) for k, v in batch.items()}
        with torch.inference_mode():
            finetuned_outputs = f_model(**batch1)

        # Process batch for f_model2
        batch2 = {k: v.to(f_model2.device) for k, v in batch.items()}
        pruned_outputs = f_model2(**batch2)

        # Calculate loss on last hidden states
        loss = top_k_mse_loss(
            pruned_outputs.logits,
            finetuned_outputs.logits.clone().to(f_model2.device).detach(),5
        )
        train_loss_list.append(loss.item())
        lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch {step+1}: Learning Rate = {lr:.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        bar.set_description(f"Train loss: {loss.item():.4f} Learning rate = {lr:.6f}")

        if step % 10 == 0:
            for bath_val in dataloader_val:
                batch1 = {k: v.to(f_model.device) for k, v in bath_val.items()}
                with torch.inference_mode():
                    finetuned_outputs = f_model(**batch1)
                # Process batch for f_model2
                    batch2 = {k: v.to(f_model2.device) for k, v in bath_val.items()}
                    pruned_outputs = f_model2(**batch2)
                loss_val = top_k_mse_loss(
                    pruned_outputs.logits,
                    finetuned_outputs.logits.clone().to(f_model2.device).detach(),5
                )
            print(f"Validation loss: {loss_val.item():.4f}")
            early_stopping(loss_val.item(), f_model2)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    test_data,test_label = get_data(args.dataset,'test')
    test_data,test_label = test_data[:100],test_label[:100]
    answers = batch_generation(f_model2,tokenizer,test_data,batch_size=10)
    pef = cal_acc_gsm8k(test_data,answers,test_label,args)
    print('the performance is {}'.format(pef))

if __name__ == "__main__":
    main()
