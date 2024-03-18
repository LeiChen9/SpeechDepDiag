import torch 
from DataProcess.data_config import DataConfig

if __name__ == "__main__":
    print(torch.cuda.is_available())
    # data_config = DataConfig('./Data/EATD-Corpus')