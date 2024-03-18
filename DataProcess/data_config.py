import os, librosa
import pandas as pd
from utils import *

class DataConfig:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_config = self.get_raw_config()
        self.empty_dic = self.get_empty_speech()
        self.name_lst, self.pos_path_lst, self.neg_path_lst, self.neutral_path_lst, self.label_lst = [], [], [], [], []
        self.data_df = self.create_data_df()
        self.dump_data(self.data_df)
        
    def check_string(self, pattern, string):
        return re.match(pattern, string) is not None

    def get_raw_config(self):
        self.data_config = {}
        for subdir in os.listdir(self.data_dir):
            if self.check_string('[a-z]_[0-9]+', subdir):
                curr_dir = os.path.join(self.data_dir, subdir)
                pos_file_name = os.path.join(curr_dir, 'positive_out.wav')
                neg_file_name = os.path.join(curr_dir, 'negative_out.wav')
                neutral_file_name = os.path.join(curr_dir, 'neutral_out.wav')
                label_file_name = os.path.join(curr_dir, 'label.txt')
                with open(label_file_name, 'r') as file:
                    label = file.read().strip()
                    label = 1 if float(label) >= 53 else 0
                self.data_config[subdir] = {
                    'pos': pos_file_name,
                    'neg': neg_file_name,
                    'neutral': neutral_file_name,
                    'label': label
                }
        return self.data_config

    def get_empty_speech(self):
        self.empty_dic = {}
        for uid, file_dic in self.data_config.items():
            for k, v in file_dic.items():
                if k == 'label':
                    continue 
                curr_time = librosa.get_duration(filename=v)
                if curr_time == 0:
                    self.empty_dic[uid] = k
        return self.empty_dic

    def create_data_df(self):
        for uid, file_dic in self.data_config.items():
            if uid in self.empty_dic.keys():
                continue
            self.name_lst.append(uid)
            self.pos_path_lst.append(file_dic['pos'])
            self.neg_path_lst.append(file_dic['neg'])
            self.neutral_path_lst.append(file_dic['neutral'])
            self.label_lst.append(file_dic['label'])

        self.data_df = pd.DataFrame({
            'name': self.name_lst,
            'pos_path': self.pos_path_lst,
            'neg_path': self.neg_path_lst,
            'neutral_path': self.neutral_path_lst,
            'label': self.label_lst
        })
        self.data_df['tag'] = self.data_df['name'].apply(lambda x: 'train' if x[0] == 't' else 'test')
        return self.data_df
    
    def dump_data(self, data_df):
        train_df = data_df[data_df['tag'] == 'train']
        test_df = data_df[data_df['tag'] == 'test']
        import pickle

        # Pickle the train_dataset dataframe
        with open('./Data/Config/train_dataset.pkl', 'wb') as f:
            pickle.dump(train_df, f)

        # Pickle the eval_dataset dataframe
        with open('./Data/Config/eval_dataset.pkl', 'wb') as f:
            pickle.dump(test_df, f)
    
# Loading the created dataset using datasets
# from datasets import load_dataset, load_metric


# data_files = {
#     "train": './train_dataset.pkl', 
#     "test": './eval_dataset.pkl'
# }

# dataset = load_dataset("pandas", data_files=data_files)
# train_dataset = dataset["train"]
# eval_dataset = dataset["test"]

# print(train_dataset)
# print(eval_dataset)

# add this __pycache__ to .gitignore