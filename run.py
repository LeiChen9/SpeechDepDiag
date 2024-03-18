import torch, pdb
from DataProcess.data_config import DataConfig
from DataProcess.DataProcessor import DataProcessor
from transformers import Wav2Vec2Model, AutoConfig

if __name__ == "__main__":
    print(torch.cuda.is_available())
    # data_config = DataConfig('./Data/EATD-Corpus')
    data_files = {
        "train": './Data/Config/train_dataset.pkl', 
        "test": './Data/Config/eval_dataset.pkl'
    }
    data_processor = DataProcessor(data_files)
    print(data_processor.train_dataset)
    print(data_processor.eval_dataset)
    
    