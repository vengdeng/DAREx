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
from fraction import Fraction
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.utils import *
from utils.getwx import *
from utils.metrics import *
from utils.analyticalq import *
from utils.deltaprune import *
from utils.outputchange import *
from vllm import LLM, SamplingParams
import copy
import numpy as np
import random
import os
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
stop_tokens = ["Q:","Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]

def objective_func(inputs,param,args):
    drop_dict = inputs['drop_dict']
    pretrained_model = inputs['pretrained_model']
    f_model = inputs['finetuned_model_2']
    perf_list = []
    seed_list = [0] if args.outputchange else [0]
    for seed in seed_list:
        print(seed)
        if seed != 0:
            torch.manual_seed(seed)
            ### you can also save the drop in advance to save time 
            drop_dict = drop(inputs['delta_param_or'],1-inputs['p'])
        if args.analytical:
            q_dict_e = optimal_q_calculation(inputs['metric_score'],inputs['delta_param'],param,p=1-inputs['p'],gpu=args.device,stop_iter=args.stop_iter)
            delta_2 = rescale_q_decoder(pretrained_model,f_model,drop_dict,q_dict_e)
        else:
            delta_2 = rescale_v_decoder(pretrained_model,f_model,drop_dict,param)

        f_model.load_state_dict(delta_2)

        # llm = LLM(model=model)
        if args.outputchange:
                inputs['pruned_model'] = f_model
                pef = -get_outpuchage_decoder(inputs)  ### add regularization
        else:
            val_data,label = inputs['val_dataset'] 
            answers = batch_generation(f_model,f_tokenizer,val_data,batch_size=10)
            pef = cal_acc_gsm8k(val_data,answers,label,args)
        perf_list.append(pef)
    # if len(perf_list) >= 3:
    #     print(perf_list)
    #     perf_list = remove_one_outlier(perf_list)
    pef = np.mean(perf_list)
    print('the performance of param {} is {}'.format(param,pef))
    return  pef

def find_optimal_param(inputs,l_bound,u_bound,step=0.1,decay=0.1):
    """
    step: step size in find the optimal q
    decay: 0-1, the decay rate for more accurate finding q, you can increase it if you need a faster finding.
    """
    best_score = -np.inf
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
    l_bound = max(0,sorted_slots[0][0] - step*(1-decay))
    u_bound = sorted_slots[0][0] + step

    best_score = -np.inf
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
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = 100000
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number_v1(completion):
    # try:
    extract_ans = completion.split('####')[1].strip()
    extract_ans = re.sub("[^0-9]", "", extract_ans)
    # except:
    #     extract_ans = None
    return extract_ans

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None
    
def cal_acc_gsm8k(gsm8k_ins,res_completions,gsm8k_answers,args):
    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        if 'Abel' in args.finetuned_model:
            try:
                y_pred = extract_answer_number_v1(completion)
            except:
                y_pred = None 
        else:
            try:
                y_pred = extract_answer_number(completion)
            except:
                y_pred = None 


        if y_pred != None:
            try:
                results.append(float(y_pred) == float(prompt_answer))
            except:
                results.append(False)
        else:
            results.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    return sum(results) / len(results)
def get_data(data_name,split):
    ins = []
    answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    ds_train = load_dataset(data_name, "main",split=split)

    for idx, item in enumerate(ds_train):
            temp_instr = problem_prompt.format(instruction=item["question"])
            ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            answers.append(temp_ans)
    #### begin with frist 200 for fast comparison
    return ins, answers
def batch_generation(f_model,f_tokenizer,gsm8k_ins,batch_size=16):
    # INVALID_ANS = "[invalid]"
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)
    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        res = f_model.generate(**f_tokenizer(prompt, return_tensors="pt",padding=True).to(f_model.device), max_new_tokens=512,temperature=0.1)
        for output in res:
            generated_text = f_tokenizer.decode(output)
            res_completions.append(generated_text)
    # print(res_completions)
    return res_completions
def batch_generation_vllm(llm,sampling_params,gsm8k_ins,batch_size=16):
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)
    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)
    return res_completions

if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--pretrain_model', type=str, default='meta-llama/Llama-2-7b-hf', help='neural network used in pretraining')
        parser.add_argument('--device', type=str, default='cpu', help='device name')
        parser.add_argument('--architect', type=str, default='bert-base-uncased', help='the architecture to be used')
        parser.add_argument('--p', type=float, default=0.01, help='the pruning rate used')
        parser.add_argument('--l_bound', type=float, default=0.1, help='the 1- pruning rate used for q_v, but analynatical different')
        parser.add_argument('--u_bound', type=float, default=0.3, help='upper bound')
        # parser.add_argument('--model_path', type=str, default='cola', help='dataset name to use')
        parser.add_argument('--dataset_name', type=str, default='openai/gsm8k', help='dataset name to use')
        parser.add_argument('--step_size', type=float, default=0.05, help='the pruning rate used')
        parser.add_argument('--decay', type=float, default=0.1, help='the pruning rate used')
        
        parser.add_argument('--finetuned_model', type=str, default=None, help='the finetuned model to use')
        parser.add_argument('--analytical', action='store_true', help='use analytical resolve q')
        parser.add_argument('--param', type=float, default=None, help='the optimal parameter')
        parser.add_argument('--stop_iter', type=float, default=0.8, help='the analytical parameter for avoid influence of outlier')
        parser.add_argument('--outputchange', action='store_true', help='use output change to resolve q')
        parser.add_argument('--inference', action='store_true', help='inference on test data')
        args = parser.parse_args()
        return args
    args = get_args()

    device = args.device
    architect = args.architect
    p=args.p
    ### you fintuned model here
    model_name = args.finetuned_model

    datasets_test_metrics = []

    data,label = get_data(args.dataset_name,'train')
    test_data,test_label = get_data(args.dataset_name,'test')
    # test_data,test_label = test_data[:100],test_label[:100]
    val_data,val_label = data[:100],label[:100]
   
    if args.inference:
        save_model_path = 'temp_path'
        llm = LLM(model=save_model_path,tensor_parallel_size=1)
        sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=512, stop=stop_tokens)
        answers = batch_generation_vllm(llm,sampling_params,test_data,batch_size=10)
        pef = cal_acc_gsm8k(test_data,answers,test_label,args)
        print(pef)
    else:
        # assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
        p_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrain_model, torch_dtype=torch.float16,device_map="cpu")
        p_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrain_model)
        
        f_tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
        f_model = AutoModelForCausalLM.from_pretrained(model_name,output_hidden_states=args.outputchange,torch_dtype=torch.float16,attn_implementation="flash_attention_2")
        
        f_model2 = AutoModelForCausalLM.from_pretrained(model_name,output_hidden_states=args.outputchange,torch_dtype=torch.float16,attn_implementation="flash_attention_2") # 
        
        f_model2.to(args.device)

        delta_param = cal_delta_param_decoder(f_model.state_dict(),p_model.state_dict())
        delta_param_or = delta_param
        inputs = {}
        ### find optimal param
        if args.analytical:
            f_model_temp = AutoModelForCausalLM.from_pretrained(model_name,output_hidden_states=True)
            tokenizer_temp = AutoTokenizer.from_pretrained(model_name)
            # data_ana,_ = get_loaders("wikitext2",tokenizer= f_tokenizer,seqlen=128,nsamples = 50)
            trainenc = get_output_data(args.dataset_name,tokenizer_temp)
            nsamples = 50
            seqlen = 128
            data_ana = output_loader(seqlen,nsamples,trainenc,tar_flag=True)
            metric_score = get_wx_decoder(f_model_temp.to(args.device),data_ana,delta_param_or,nsamples,seqlen)
            del f_model_temp
            del tokenizer_temp
            # f_model = AutoModelForCausalLM.from_pretrained(model_name,output_hidden_states=args.outputchange)
            
        if args.outputchange:
            #### change the dataset to what you want #####
            inputs['delta_param_or'] = delta_param_or
            trainenc = get_output_data(args.dataset_name,f_tokenizer)
            seqlen = 256
            nsamples = 100
            dataloader = output_loader(seqlen,nsamples,trainenc)
            with torch.no_grad():
                f_model.to(args.device)
                f_model.eval()
                ##### finetuned_model.roberta for roberta model
                # output = f_model.model.forward(torch.cat(dataloader).to(args.device))
                # last_layer_mean_o = output.hidden_states[-1]
                output = f_model(torch.cat(dataloader).to(args.device))
                last_layer_mean_o = output.logits
            del f_model
            inputs['last_layer_mean_o'] = last_layer_mean_o.to(args.device)
            inputs['dataloader'] = dataloader

        if args.param is None:
            drop_dict = drop_decoder(delta_param_or,1-p)
            if args.analytical:
                inputs['metric_score'] = metric_score
            inputs['train_dataset'] = [data,label]
            inputs['val_dataset'] = [val_data,val_label]
            inputs['tokenizer'] = f_tokenizer
            inputs['finetuned_model_2'] = f_model2
            inputs['drop_dict'] = drop_dict
            inputs['pretrained_model'] = p_model
            inputs['delta_param'] = delta_param
            inputs['p'] = p
            param = find_optimal_param(inputs,args.l_bound,args.u_bound,step=args.step_size,decay=args.decay)
        else:
            param = args.param

        if args.analytical:
            q_dict_e = optimal_q_calculation(metric_score,delta_param,param,p=1-p,gpu=args.device,stop_iter=args.stop_iter)
        else:
            #### the validaiton q here
            q = param

        res_all = {}
        res_all[p] = []
        ##### drop for 
        for seed in [0]:
            set_seed(seed)
            drop_dict = drop(delta_param_or,1-p)
            if args.analytical:
                delta_2 = rescale_q_decoder(p_model,f_model2,drop_dict,q_dict_e)
            else:
                delta_2 = rescale_v_decoder(p_model,f_model2,drop_dict,param)

            # new_dict2 = add_extra_back(f_model_c,delta_2)
            # if args.outputchange:
            #     f_model2 = AutoModelForCausalLM.from_pretrained('GAIR/Abel-7B-001')
            #     f_model2.to(args.device)
            f_model2.load_state_dict(delta_2)
            save_model_path = 'temp_path'
            os.makedirs(save_model_path, exist_ok=True)
            f_model2.save_pretrained(save_directory=save_model_path)
            f_tokenizer.save_pretrained(save_directory=save_model_path)

            llm = LLM(model=save_model_path,tensor_parallel_size=1)
            sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=800, stop=stop_tokens)
            answers = batch_generation_vllm(llm,sampling_params,test_data,batch_size=16)
            pef = cal_acc_gsm8k(test_data,answers,test_label,args)

            res_all[p].append(pef)
        print(np.mean(res_all[p]))