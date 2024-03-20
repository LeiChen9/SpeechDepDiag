from transformers import AutoConfig, Wav2Vec2Processor
import torch
import numpy as np
import torchaudio
import transformers
import pdb
from datasets import load_dataset, load_metric

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class DataProcessor:
    def __init__(self, data_files):
        self.pretrain_model_setup()
        dataset = load_dataset("pandas", data_files=data_files)
        self.train_dataset = dataset["train"].map(
            self.preprocess_function,
            batch_size=1,
            batched=True,
            num_proc=4,
        )
        self.eval_dataset = dataset["test"].map(
            self.preprocess_function,
            batch_size=1,
            batched=True,
            num_proc=4,
        )
    
    def pretrain_model_setup(self):
        model_name_or_path = "facebook/wav2vec2-base-960h"
        pooling_mode = "mean"
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=2,
            label2id={'depression': 1, 'health': 0},
            id2label={0: 'health', 1: 'depression'},
            finetuning_task="wav2vec2_clf",
        )
        setattr(config, 'pooling_mode', pooling_mode)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        print(f"The target sampling rate: {self.target_sampling_rate}")
        
    def preprocess_function(self, examples):
        # pos_list = [self.speech_file_to_array_fn(path) for path in examples['pos_path']]
        # neg_list = [self.speech_file_to_array_fn(path) for path in examples['neg_path']]
        # neu_list = [self.speech_file_to_array_fn(path) for path in examples['neutral_path']]
        speech_list = [self.speech_file_to_array_fn(path) for path in examples['path']]
        target_list = examples['label']
        
        # result = transformers.feature_extraction_utils.BatchFeature()
        # result['pos'] = self.processor(pos_list, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding="longest")['input_values'].numpy()
        # result['neg'] = self.processor(neg_list, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding="longest")['input_values'].numpy()
        # result['neu'] = self.processor(neu_list, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding="longest")['input_values'].numpy()
        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        result["labels"] = target_list

        return result

    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

if __name__ == "__main__":
    # data_files = {
    #     "train": './Data/Config/train_dataset.pkl', 
    #     "test": './Data/Config/eval_dataset.pkl'
    # }
    # data_processor = DataProcessor(data_files)
    # print(data_processor.train_dataset)
    # print(data_processor.eval_dataset)
    example = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    pos_emb = sinusoids(example.shape[0], )
    pdb.set_trace()