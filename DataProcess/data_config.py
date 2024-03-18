import os, librosa
from utils import *

data_config = {}
data_dir = './Data/EATD-Corpus/'
data_df = []
for subdir in os.listdir(data_dir):
    if check_string('[a-z]_[0-9]+', subdir):
        curr_dir = os.path.join(data_dir, subdir)
        pos_file_name = os.path.join(curr_dir, 'positive_out.wav')
        neg_file_name = os.path.join(curr_dir, 'negative_out.wav')
        neutral_file_name = os.path.join(curr_dir, 'neutral_out.wav')
        label_file_name = os.path.join(curr_dir, 'label.txt')
        with open(label_file_name, 'r') as file:
            label = file.read().strip()
            label = 1 if float(label) >= 53 else 0
        data_config[subdir] = {
            'pos': pos_file_name,
            'neg': neg_file_name,
            'neutral': neutral_file_name,
            'label': label
        }
# print("Getting rid of empty speech...")
empty_dic = {}
for uid, file_dic in data_config.items():
    for k, v in file_dic.items():
        if k == 'label':
            continue 
        curr_time = librosa.get_duration(filename=v)
        if curr_time == 0:
            empty_dic[uid] = k
# for uid, k in empty_dic.items():    
#     print("{}-{}-{}".format(uid, k, data_config[uid].pop(k, None)))

name_lst, pos_path_lst, neg_path_lst, neutral_path_lst, label_lst = [], [], [], [], []
for uid, file_dic in data_config.items():
    if uid in empty_dic.keys():
        continue
    name_lst.append(uid)
    pos_path_lst.append(file_dic['pos'])
    neg_path_lst.append(file_dic['neg'])
    neutral_path_lst.append(file_dic['neutral'])
    label_lst.append(file_dic['label'])

data_df = pd.DataFrame({
    'name': name_lst,
    'pos_path': pos_path_lst,
    'neg_path': neg_path_lst,
    'neutral_path': neutral_path_lst,
    'label': label_lst
})

data_df['tag'] = data_df['name'].apply(lambda x: 'train' if x[0] == 't' else 'test')

data_df.head(5)

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