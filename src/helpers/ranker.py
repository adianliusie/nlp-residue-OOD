import random

from typing import List
from types import SimpleNamespace
from copy import deepcopy

from abc import ABC, abstractmethod

def make_ranker(ranker_name:str, *args):
    if ranker_name=='random':
        return RandomRanker(*args)
    elif ranker_name=='length':
        return LengthRanker()    
    else:
        raise ValueError("invalid ranking option")
    
class Ranker(ABC):
    """ base class for all rankers """
    def __init__(self):
        pass
    
    def rank_data(self, data:List)->List:
        data_copy = deepcopy(data)
        data_copy = sorted(data_copy, key=lambda x: self.get_ex_score(x), reverse=True)
        return data_copy
        
    @abstractmethod
    def get_ex_score(self, ex:SimpleNamespace)->float:
        pass
    
    def __call__(self, *args, **kwargs):
        return self.rank_data(*args, **kwargs)
        
class LengthRanker(Ranker):
    """ ranks all examples based on length of ids """
    def get_ex_score(self, ex:SimpleNamespace)->float:
        return len(ex.ids)

class RandomRanker(Ranker):
    """ ranks all examples in a random order based on the seed """
    def __init__(self, seed_num:int=1):
        random.seed(seed_num)
        
    def get_ex_score(self, ex:SimpleNamespace)->float:
        return random.random()
