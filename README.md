# DARE the Extreme: Revisiting Delta-Parameter Pruning For Fine-Tuned Models

[![Conference](https://img.shields.io/badge/ICLR-2025-blue.svg)](https://iclr.cc/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Arxiv](https://img.shields.io/badge/ArXiv-2502.19980-b31b1b.svg)](https://arxiv.org/abs/2410.09344)

📌 **Official repository for our ICLR 2025 Spotlight paper:**

> **DARE the Extreme: Revisiting Delta-Parameter Pruning For Fine-Tuned Models**  
> * [Wenlong Deng](https://vengdeng.github.io/), Yize Zhao,  Vala Vakilian, [Minghui Chen](https://chenminghui.com), [Xiaoxiao Li](https://tea.ece.ubc.ca/), [Christos Thrampoulidi](https://sites.google.com/view/cthrampo)*  
> 📄 [[Arxiv](https://arxiv.org/abs/2410.09344)]  

## 📌 Abstract
Storing open-source fine-tuned models separately introduces redundancy and increases response times
in applications utilizing multiple models. Delta-parameter pruning (DPP), particularly the random drop
and rescale (DARE) method proposed by Yu et al., addresses this by pruning the majority of delta parameters—the differences between fine-tuned and pre-trained model weights—while typically maintaining minimal performance loss. However, DARE fails when either the pruning rate or the magnitude of the delta parameters is large. We highlight two key reasons for this failure: (1) an excessively large rescaling factor
as pruning rates increase, and (2) high mean and variance in the delta parameters. To push DARE’s limits,
we introduce DAREx (DARE the eXtreme), which features two algorithmic improvements: (1) DAREx-q,
a rescaling factor modification that significantly boosts performance at high pruning rates (e.g., > 30%
on COLA and SST2 for encoder models, with even greater gains in decoder models), and (2) DAREx-L2,
which combines DARE with AdamR, an in-training method that applies appropriate delta regularization before DPP. We also demonstrate that DAREx-q can be seamlessly combined with vanilla parameter-efficient
fine-tuning techniques like LoRA and can facilitate structural DPP. Additionally, we revisit the application
of importance-based pruning techniques within DPP, demonstrating that they outperform random-based
methods when delta parameters are large. Through this comprehensive study, we develop a pipeline for
selecting the most appropriate DPP method under various practical scenarios

## Find q for encoder

For encoder model, we share the fintuned models in [link](https://drive.google.com/drive/folders/1A3YkK1iGoj2DyvJ7LhufV2fbaV4g70Rc?usp=sharing)

```
python find_q_encoder.py --l_bound 0.1 --u_bound 0.5 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model cola model  --dataset_name cola  --param 0.37

python find_q_encoder.py --l_bound 0.3 --u_bound 0.7 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model sst2 model  --dataset_name sst2 --param 1.01

python find_q_encoder.py --l_bound 0.3 --u_bound 0.7 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model mrpc model  --dataset_name mrpc --param 0.27

python find_q_encoder.py --l_bound 0.3 --u_bound 0.7 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model stsb model  --dataset_name stsb --param 0.36

```

(1) Remove the --param if you want to find your own

(2) Remove --analytical if you want to find global q, and set the l_bound=1-p(pruning rate) and u_bound <= 1.0

(3) add --outputchange if you want to unsupervised way

here is an example:

```
python find_q_encoder.py --l_bound 0.01 --u_bound 0.1 --step_size 0.05  --device cuda:1   --finetuned_model mrpc model  --dataset_name mrpc -
```


## Find q for decoder model

We have conducted an speedup version of finding q for decoder model by leveraging bf16 and flash attention. As state in our paper, we recommend to find global q for decoder models. Additionally, we provide a per-layer optimization strategy for usage.

For decoder model, we provide an example with 🤗 <a href="https://huggingface.co/GAIR/GAIRMath-Abel-7b" target="_blank"> Abel-7B-001 </a>. 

```
find_q_decoder.py --l_bound 0.01 --u_bound 0.1 --step_size 0.05 --p 0.01  --device cuda:2 --finetuned_model GAIR/Abel-7B-001 
```

We also provide a optimization-based strategy to find per-layer q, this is an initial version, you need to pay attention to the early stop. Here is an example with Meta-Math fintuned Qwen-05B [link](https://drive.google.com/drive/folders/1TOeZ3fuW78eqZKWs3ePUY0R5xXbWmO-X?usp=drive_link)

```
python perlayer_opt.py
```

## 📜 Citation

If you find our work useful and relevant, please cite:

```bibtex
@article{deng2024dare,
  title={DARE the Extreme: Revisiting Delta-Parameter Pruning For Fine-Tuned Models},
  author={Deng, Wenlong and Zhao, Yize and Vakilian, Vala and Chen, Minghui and Li, Xiaoxiao and Thrampoulidis, Christos},
  journal={The Thirteenth International Conference on Learning Representations},
}
