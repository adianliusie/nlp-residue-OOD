import random

from tqdm import tqdm 
from copy import deepcopy
from typing import List, Dict, Tuple
from datasets import load_dataset

### Main Data Loading Method #############################################################
def load_data(data_name:str, lim:int=None)->Tuple['train', 'dev', 'test']:
    if   data_name == 'imdb':   train, dev, test = _load_imdb()
    elif data_name == 'yelp':   train, dev, test = _load_yelp()
    elif data_name == 'boolq':  train, dev, test = _load_boolq()
    elif data_name == 'paws':   train, dev, test = _load_paws()

    else: raise ValueError('invalid dataset provided')
        
    if lim:
        train, dev, test = train[:lim], dev[:lim], test[:lim]
    
    return train, dev, test

### Individual Data set Loader Functions #################################################
def _load_imdb()->List[Dict['text', 'label']]:
    dataset = load_dataset("imdb")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])
    return train, dev, test

def _load_yelp():
    dataset = load_dataset("yelp_review_full")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])
    return train, dev, test

def _load_boolq()->List[Dict['text', 'label']]:
    dataset = load_dataset("super_glue", "boolq")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test   = list(dataset['validation'])
    train, dev, test = _rename_keys(train, dev, test, old_key='passage', new_key='text')
    return train, dev, test

def _load_paws()->List[Dict['text_1, text_2', 'label']]:
    dataset = load_dataset("paws", 'labeled_final')
    train = list(dataset['train'])
    dev   = list(dataset['validation'])
    test  = list(dataset['test'])
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence1', new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence2', new_key='text_2')
    return train, dev, test

def _load_qqp()->List[Dict['text_1, text_2', 'label']]:
    dataset = load_dataset("quora")
    train_data = list(dataset['train'])
    train_data, test = _create_splits(train_data, 0.9)
    train, dev       = _create_splits(train_data, 0.8)
    train, dev, test = [[qqp_split(ex) for ex in data] for data in (train, dev, test)]
    return train, dev, test

### Helper Methods for processing the data sets #########################################
def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random.seed(1)
    random.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

def _rename_keys(train:list, dev:list, test:list, old_key:str, new_key:str):
    train = [_rename_key(ex, old_key, new_key) for ex in train]
    dev   = [_rename_key(ex, old_key, new_key) for ex in dev]
    test  = [_rename_key(ex, old_key, new_key) for ex in test]
    return train, dev, test

def _rename_key(ex:dict, old_key:str='content', new_key:str='text'):
    """ convert key name from the old_key to 'text' """
    ex = ex.copy()
    ex[new_key] = ex.pop(old_key)
    return ex

def qqp_split(ex:dict):
    ex = ex.copy()
    ex['text_1'], ex['text_2'] = ex.pop('text')
    return ex
