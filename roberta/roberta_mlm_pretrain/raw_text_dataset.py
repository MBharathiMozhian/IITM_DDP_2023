import os

import numpy as np
import torch
# from nlp import load_dataset
from datasets import load_dataset
from transformers import LineByLineTextDataset
from torch.utils.data import Dataset


class RawTextDataset(Dataset):
    """
    Custom Torch Dataset for tokenizing large (upto 100,000,000+ sequences) text corpus by not loading the entire dataset into cache
    """ 
    
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        data_files = get_data_files(file_path)
        # data_files = get_data_files('/scratch/scratch6/bharathimozhian/bpe_pubchem112m/datasets/final/')
        # data_dir = '/scratch/scratch6/bharathimozhian/bpe_pubchem112m/datasets/final'
        # cache_dir = '/scratch/scratch6/bharathimozhian/bpe_pubchem112m/datasets/cache'
        # print(cache_dir)
        # print(file_path)

        # self.dataset = load_dataset("text", data_dir=data_dir, cache_dir=cache_dir)['train']
        self.dataset = load_dataset("text", data_files=data_files)['train']
        print('Loaded Dataset')
        self.len = len(self.dataset)
        print('Number of lines: ' + str(self.len))
        print('Block size: ' + str(self.block_size))

    def __len__(self):
        return self.len

    def preprocess(self, feature_dict):
        batch_encoding = self.tokenizer(
            feature_dict['text'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.block_size,
        )
        return torch.tensor(batch_encoding['input_ids']) # while running in GPU
        # device = torch.device('cpu')
        # return torch.tensor(batch_encoding['input_ids']).to(device) # while running in CPU

    def __getitem__(self, i):
        line = self.dataset[i]
        example = self.preprocess(line)
        return example


def get_data_files(train_path):
    if os.path.isdir(train_path):
        return [
            os.path.join(train_path, file_name) for file_name in os.listdir(train_path)
        ]
    elif os.path.isfile(train_path):
        return train_path
    
    raise ValueError('Please pass in a proper train path')