import random 
import pickle 
import os

from types import SimpleNamespace
from tqdm import tqdm


from ..utils.data_utils  import load_data
from ..utils.torch_utils import load_tokenizer

class DataLoader:
    def __init__(self, trans_name:str):
        self.tokenizer = load_tokenizer(trans_name)

    def prep_single(self, data:list)->list:        
        output = []
        for ex in tqdm(data):            
            text = ex['text']
            ids = self.tokenizer(ex['text']).input_ids
            label = ex['label']
            output.append(SimpleNamespace(text=text, ids=ids, label=label))
        return output
    
    def prep_double(self, data:list)->list:
        output = []
        for ex in tqdm(data):
            text = (ex['text_1'], ex['text_2'])
            ids_1  = self.tokenizer(ex['text_1']).input_ids
            ids_2  = self.tokenizer(ex['text_2']).input_ids
            ids = ids_1 + ids_2[1:]
            label = ex['label']
            output.append(SimpleNamespace(text=text, ids=ids, label=label))
        return output
            
    def get_data(self, data_name:str, lim:int=None):
        train, dev, test = load_data(data_name, lim)
        train, dev, test = [self.process_ids(split) for split in (train, dev, test)]
        return train, dev, test
    
    def get_data_split(self, data_name:str, split:str, lim:int=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = load_data(data_name, lim)[split_index[split]]
        data = self.process_ids(data)
        return data
    
    def __call__(self, *args, **kwargs):
        return self.get_data(*args, **kwargs)

    def process_ids(self, data):
        if 'text' in data[0]:
            output = self.prep_single(data)
        elif 'text_1' in data[0] and 'text_2' in data[0]:
            output = self.prep_double(data)
        return output
    
