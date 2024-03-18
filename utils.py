import re, random 
import numpy as np
import torch

def check_string(re_exp, str):
    res = re.search(re_exp, str)
    if res:
        return True 
    else:
        return False
    
def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)