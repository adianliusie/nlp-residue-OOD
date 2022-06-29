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

    def prep_split(self, data:list):
        random.seed(1) #set random seed for reproducibility
        
        output = []
        for ex in tqdm(data):
            text  = ex['text'].lower()
            label = ex['label']
            
            ids = self.tokenizer(text).input_ids
            output.append(SimpleNamespace(text=text, ids=ids, label=label))
        return output
                                                 
    def get_data(self, data_name:str, lim:int=None):
        train, dev, test = load_data(data_name, lim)
        train, dev, test = [self.prep_split(split) for split in (train, dev, test)]
        return train, dev, test
    
    def get_data_split(self, data_name:str, split:str, lim:int=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = load_data(data_name)[split_index[split]]
        data = self.prep_split(data)
        return data
    
    def __call__(self, *args, **kwargs):
        return self.get_data(*args, **kwargs)
