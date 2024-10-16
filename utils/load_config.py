import torch

if torch.cuda.is_available():
    cache_dir = ""
else:
    cache_dir = ""
