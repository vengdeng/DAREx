import torch
import numpy as np
import random
from datasets import load_dataset
from utils.getwx import *
from utils.utils import *
### load dataset 
def get_outpuchage(inputs,args):
    finetuned_model = inputs['pruned_model']
    finetuned_model.eval()
    dataloader = inputs['dataloader']
    last_layer_mean_o = inputs['last_layer_mean_o']
    with torch.no_grad():
        if 'roberta' in args.architect:
            last_layer_mean_p = finetuned_model.roberta(torch.cat(dataloader).to(finetuned_model.device)).last_hidden_state
            last_layer_mean_p = finetuned_model.classifier.dense(last_layer_mean_p)
        elif 'bert' in args.architect:
            last_layer_mean_p = finetuned_model.bert(torch.cat(dataloader).to(finetuned_model.device)).last_hidden_state
        else:
            print('add your own last_hidden_state code ')
            assert 1 == 2
    mean_diff = torch.mean(torch.abs(last_layer_mean_o - last_layer_mean_p))
    return mean_diff.cpu()



def get_outpuchage_decoder(inputs):
    finetuned_model = inputs['pruned_model']
    finetuned_model.eval()
    dataloader = inputs['dataloader']
    last_layer_mean_o = inputs['last_layer_mean_o']
    with torch.no_grad():
        # output = finetuned_model.model.forward(torch.cat(dataloader).to(finetuned_model.device))
        # last_layer_mean_p = output.hidden_states[-1]
        output = finetuned_model(torch.cat(dataloader).to(finetuned_model.device))
        last_layer_mean_p = output.logits
    # mean_diff = torch.mean(torch.abs(last_layer_mean_o - last_layer_mean_p))
    mean_diff = top_k_mse_loss(
            last_layer_mean_p,
            last_layer_mean_o.clone().detach(),5
        )
    return mean_diff.cpu()

def output_loader(seqlen,nsamples,trainenc,tar_flag=False):
    random.seed(0)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        if not tar_flag:
            trainloader.append(inp)
        else:
            trainloader.append((inp, tar))
    return trainloader

def get_output_data(dataset_name,tokenizer):
    if dataset_name in ['sst2','cola','mrpc','stsb']:
        dataset = load_dataset('glue', dataset_name, split='train')
        try:
            trainenc = tokenizer(" ".join(dataset['sentence']), return_tensors='pt')
        except:
            trainenc = tokenizer(" ".join(dataset['sentence1']), return_tensors='pt')
    elif 'gsm8k' in dataset_name:
        dataset = load_dataset("openai/gsm8k", "main",split='test')
        # INVALID_ANS = "[invalid]"
        gsm8k_answers_few_shot = []
        # problem_prompt = (
        #     "Below is an instruction that describes a task. "
        #     "Write a response that appropriately completes the request.\n\n"
        #     "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        # )
        for idx, item in enumerate(dataset):
            example = item["question"] + item["answer"]
            # example = problem_prompt.format(instruction=item["question"]) + item["answer"]
            gsm8k_answers_few_shot.append(example)
            if idx == 100:
                break
        trainenc = tokenizer(" ".join(gsm8k_answers_few_shot), return_tensors='pt')
    else:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    return trainenc