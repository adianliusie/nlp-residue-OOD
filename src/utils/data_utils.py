import random

from tqdm import tqdm 
from copy import deepcopy
from typing import List, Dict, Tuple
from datasets import load_dataset
from functools import lru_cache

### Main Data Loading Method #############################################################
@lru_cache(maxsize = 5)
def load_data(data_name:str, lim:int=None)->Tuple['train', 'dev', 'test']:
    print(f'############ {data_name} ############')

    # If biased enabled, create a length bias in the data
    if data_name.endswith('_biased'):
        data_name = data_name.replace('_biased', '')
        train, dev, test = create_len_bias(data_name)
    
    # If multiple datasets given, use multitask learning
    elif '_' in data_name:
        train, dev, test = join_multiple_datasets(data_name)
    
    # Otherwise load the standard data
    elif data_name == 'imdb':   train, dev, test = _load_imdb()
    elif data_name == 'rt':     train, dev, test = _load_rotten_tomatoes()
    elif data_name == 'sst':    train, dev, test = _load_sst()
    elif data_name == 'sst2':   train, dev, test = _load_sst2()
    elif data_name == 'yelp':   train, dev, test = _load_yelp()
    elif data_name == 'boolq':  train, dev, test = _load_boolq()
    elif data_name == 'paws':   train, dev, test = _load_paws()
    elif data_name == 'snli':   train, dev, test = _load_snli()
    elif data_name == 'mnli':   train, dev, test = _load_mnli()
    elif data_name == 'hans':   train, dev, test = _load_hans()
    else: raise ValueError('invalid dataset provided')
        
    # For debugging, can return a subset of the data
    if lim:
        train = get_data_sample(train, lim)
        dev   = get_data_sample(dev, lim)
        test  = get_data_sample(test, lim)
        
    return train, dev, test

def flatten(x):
    return [i for j in x for i in j]

def join_multiple_datasets(data_name):
    data_names = data_name.split('_')
    data_sets = [load_data(data_name) for data_name in data_names]
    train, dev, test = [flatten(data) for data in zip(*data_sets)]
    return train[:lim], dev[:lim], test[:lim]
    
def create_len_bias(data_name):
    train, dev, test = load_data(data_name)
    train, dev = [_class_divide(split) for split in [train, dev]]
    return train, dev, test

def _class_divide(data_split):
    labels = set([x['label'] for x in data_split])
    
    # Split data set by classes, and order by len
    classes_split = {}
    for label in labels:
        class_split = [x for x in data_split if x['label'] == label]
        classes_split[label] = sorted(class_split, key=lambda x: len(x['text']))
    
    # introduce natural length shortcuts in data
    output = []
    for k, class_examples in classes_split.items():
        ex_range = len(class_examples)/len(labels)
        r1, r2 = int(k*ex_range), int((k+1)*ex_range)
        selected_ex = class_examples[r1:r2]
        output += selected_ex
        
    return output

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
    def qqp_split(ex:dict):
        ex = ex.copy()
        ex['text_1'], ex['text_2'] = ex.pop('text')
        return ex
    
    dataset = load_dataset("quora")
    train_data = list(dataset['train'])
    train_data, test = _create_splits(train_data, 0.9)
    train, dev       = _create_splits(train_data, 0.8)
    train, dev, test = [[qqp_split(ex) for ex in data] for data in (train, dev, test)]
    return train, dev, test

def _load_rotten_tomatoes():
    dataset = load_dataset("rotten_tomatoes")
    train = list(dataset['train'])
    dev   = list(dataset['validation'])
    test  = list(dataset['test'])
    return train, dev, test

def _load_sst():
    dataset = load_dataset('glue', 'sst2')
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])
    
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence', new_key='text')
    return train, dev, test

def _load_hans()->List[Dict['text_1', 'label']]:
    dataset = load_dataset("hans")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test  = list(dataset['validation'])
    
    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

def _load_snli():
    def _filter(data_split):
        return [i for i in data_split if i['label'] != -1]
        
    dataset = load_dataset("snli")
    train = list(dataset['train'])
    train = get_data_sample(train, 50_000)
    dev   = list(dataset['validation'])
    test  = list(dataset['test'])
    
    train, dev, test = [_filter(data) for data in [train, dev, test]]
    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

def _load_mnli():
    dataset = load_dataset('glue', 'mnli')
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation_matched'])
    
    train = get_data_sample(train, 50_000)
    dev   = get_data_sample(dev, 10_000)

    train, dev, test = _rename_keys(train, dev, test, old_key='premise',    new_key='text_1')
    train, dev, test = _rename_keys(train, dev, test, old_key='hypothesis', new_key='text_2')
    return train, dev, test

### Helper Methods for processing the data sets #########################################
def get_data_sample(examples:list, lim:int):
    examples = deepcopy(examples)
    random.seed(1)
    random.shuffle(examples)
    return examples[:lim]
    
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

"""
old method that splits using median

def _class_divide(data_split):
    labels = set([x['label'] for x in data_split])
    
    # Split data set by classes, and order by len
    classes_split = {}
    for label in labels:
        class_split = [x for x in data_split if x['label'] == label]
        classes_split[label] = sorted(class_split, key=lambda x: len(x['text']))
    
    #calculate median length per class
    class_medians = []
    for label, examples in classes_split.items():
        median_ex = examples[(len(examples)-1)//2]
        median_len = len(median_ex['text'])
        class_medians.append((label, median_len))
        
    # introduce natural length shortcuts in data
    output = []
    sorted_medians = sorted(class_medians, key=lambda x:x[1])
    for k, (label, _) in enumerate(sorted_medians):
        class_examples = classes_split[label]
        ex_range = len(class_examples)/len(labels)
        r1, r2 = int(k*ex_range), int((k+1)*ex_range)
        selected_ex = class_examples[r1:r2]
        output += selected_ex
        
    return output
"""