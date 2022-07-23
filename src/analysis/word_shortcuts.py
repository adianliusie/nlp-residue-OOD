### TEMP for debugging- delete cell 
import re
import numpy as np
from collections import defaultdict
from typing import List, Tuple

from src.utils.data_utils import load_data

def get_word_counts(data:List[list])->dict:
    """ returns counts of words in a data set"""
    word_counts = defaultdict(lambda:0)
    for ex in data:
        text = ex['text'].lower()
        words = re.findall(r"[\w']+|[.,!?;]", text)
        word_set = set(words)
        for word in word_set:
            word_counts[word] += 1
    return word_counts

def get_vocab(word_counts:dict, thresh:int=100)->dict:
    """ truncates vocab to those with above thresh occurences"""
    vocab = {word:count for word, count in word_counts.items() if count > thresh}
    return vocab 

def get_vocab_occurences(data, vocab:dict)->list:
    """ returns the ids of all examples which contains the word"""
    word_occurences = {k:[] for k in vocab}
    for k, ex in enumerate(data):
        for word in vocab:
            if word in ex['text'].lower():
                word_occurences[word].append(k)
    return word_occurences

def word_bias(data, vocab, num_labels:int=2)->list:
    """ returns occurences of each word in vocab"""
    vocab = set([k for k in vocab])
    vocab_counts = defaultdict(lambda: np.zeros(num_labels))

    for ex in data:
        text = ex['text'].lower()
        words = re.findall(r"[\w']+|[.,!?;]", text)
        label = ex['label']
        
        for word in set(words).intersection(vocab):
            vocab_counts[word][label] += 1
    return vocab_counts


