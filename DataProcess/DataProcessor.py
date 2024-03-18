from transformers import AutoConfig, Wav2Vec2Processor
from transformers import Wav2Vec2Model
import torchaudio
import transformers
from datasets import load_dataset, load_metric

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
        pos_list = [self.speech_file_to_array_fn(path) for path in examples['pos_path']]
        neg_list = [self.speech_file_to_array_fn(path) for path in examples['neg_path']]
        neu_list = [self.speech_file_to_array_fn(path) for path in examples['neutral_path']]
        target_list = examples['label']
        
        result = transformers.feature_extraction_utils.BatchFeature()
        result['pos'] = self.processor(pos_list, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding="longest")['input_values'].numpy()
        result['neg'] = self.processor(neg_list, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding="longest")['input_values'].numpy()
        result['neu'] = self.processor(neu_list, sampling_rate=self.target_sampling_rate, return_tensors="pt", padding="longest")['input_values'].numpy()
        result["labels"] = target_list

        return result

    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

if __name__ == "__main__":
    data_files = {
        "train": './Data/Config/train_dataset.pkl', 
        "test": './Data/Config/eval_dataset.pkl'
    }
    data_processor = DataProcessor(data_files)
    print(data_processor.train_dataset)
    print(data_processor.eval_dataset)