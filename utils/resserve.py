import torch
import gc
import torch.nn as nn

def res_diff(base_model,finemodel, delta_dict,rate):
    def compress_submodule(name, subname, module, submodule):
        target_device = submodule.weight.device
                    
        base_weight = base_model.get_submodule(f"{name}.{subname}").weight.detach().to(target_device)
        ### get the finetuned model's bias if exists
        try :
            f_bias = finemodel.get_submodule(f"{name}.{subname}").bias.detach().to(target_device)
        except:
            f_bias = None
        finetuned_weight = delta_dict[f"{name}.{subname}" + '.weight'].detach().to(target_device)

        compressed = BinaryDiff(
            base=base_weight,
            delta=finetuned_weight,
            bias = f_bias,
            rate = rate
        ).to(target_device)

        del submodule, base_weight
        setattr(module, subname, None)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(module, subname, compressed)

    # TODO: this can be parallelized
    for name, module in finemodel.named_modules():
        if "mlp" in name or "self_attn" in name:
            for subname, submodule in module.named_children():
                if name + '.' + subname + '.weight' in delta_dict.keys():
                    compress_submodule(name, subname, module, submodule)
class BinaryDiff(nn.Module):
    def __init__(self, base, delta,bias=None,rate=1):
        super().__init__()
        self.register_buffer("diff", delta.to_sparse())
        self.register_buffer("base", base.T)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias",  torch.tensor(0))
        self.register_buffer("rate_base", torch.tensor(rate,dtype=torch.float32,requires_grad=False))
        self.relu = nn.ReLU()
        self.register_parameter(
            "coeff",
            nn.Parameter(
                torch.tensor(
                    0.0001,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=base.device,
                )
            ),
        )
        del base, delta

    def forward(self, x):
        
        batch_size, seq_len, feature_size = x.size()
        ##### TODO further speedup with pytorch_sparse
        ### use torch.abs(self.coeff) as we know we should increase the q.
        y = x @ self.base +  torch.sparse.mm(self.diff, x.reshape(-1, feature_size).T).T.reshape(batch_size, seq_len,-1)/ (torch.abs(self.coeff)+self.rate_base) + self.bias
        return y
    
def preprocess(tokenizer, examples, max_length=128):
    return tokenizer(
        examples["question"]+examples["answer"], padding="max_length", truncation=True, max_length=max_length
    )
 
