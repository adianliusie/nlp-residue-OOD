import random
import torch

import torch.nn.functional as F
from typing import List
from types import SimpleNamespace
from copy import deepcopy

from abc import ABC, abstractmethod

from .dir_helper import DirHelper
from ..models import select_model
from ..utils.torch_utils import no_grad
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def make_ranker(ranker_name:str, *args):
    if ranker_name == 'random':
        return RandomPruner(*args)
    elif ranker_name == 'length':
        return LengthPruner() 
    elif ranker_name == 'loss':
        return LossPruner()
    elif ranker_name == 'kmeans':
        return KMeansPruner()
    else:
        raise ValueError("invalid ranking option")

### Super DataPruner Classes #########################################################
class DataPruner(ABC):
    """ base class for all rankers """
    def __init__(self, seed_num:int=None):
        if seed_num:
            random.seed(seed_num)
    
    def rank_data(self, data:List)->List:
        data_copy = deepcopy(data)
        data_copy = sorted(data_copy, key=lambda x: self.get_ex_score(x), reverse=True)
        return data_copy
    
    def get_ex_score(self, ex:SimpleNamespace)->float:
        pass
    
    def filter_data(self, data:List, ret_frac:float)->List:
        N = int(ret_frac*len(data))
        data = self.rank_data(data)
        return data[:N]   

    def __call__(self, *args, **kwargs):
        return self.filter_data(*args, **kwargs)

class ModelDataPruner(DataPruner, ABC):
    def load_model(self, exp_path):
        dir_ = DirHelper.load_dir(exp_path)
        args = dir_.load_args('model_args.json')
        
        self.model = select_model(model_name=args.transformer)
        self.model.load_state_dict(
            torch.load(dir_.abs_path + f'/models/base.pt'))
        
        self.max_len = args.max_len
        self.model.to('cuda')
        
    @no_grad
    def model_output(self, ex)->SimpleNamespace:
        ids = torch.LongTensor(ex.ids[:self.max_len]).unsqueeze(0)
        ids = ids.to('cuda')
        return self.model(ids)
    
    def filter_data(self, *args, **kwargs)->List:
        """ overwriting parent to free model gpu memory after use """
        output = super().filter_data(*args, **kwargs)
        self.model.to('cpu')
        torch.cuda.empty_cache() 
        return output
    
### Basic Rankers ###############################################################
class LengthPruner(DataPruner):
    """ ranks all examples based on length of ids """
    def get_ex_score(self, ex:SimpleNamespace)->int:
        return len(ex.ids)

class RandomPruner(DataPruner):
    """ ranks all examples in a random order based on the seed """
    def get_ex_score(self, ex:SimpleNamespace)->float:
        return random.random()

### Model Based Rankers #########################################################

class LossPruner(ModelDataPruner):
    """ ranks all examples based on the loss of a model trained already on the examples """
    def __init__(self, seed_num:int=1):
        super().__init__(seed_num)
        model_path = '/home/alta/Conversational/OET/al826/2022/shortcuts/data_pruning/trained_models/boolq/baseline/0'   
        self.load_model(model_path)
     
    def get_ex_score(self, ex)->float:
        label = torch.LongTensor([ex.label]).to('cuda')
        y = self.model_output(ex).y
        loss = F.cross_entropy(y, label).item()
        return float(loss)
        
class KMeansPruner(ModelDataPruner):
    '''
        View samples in encoder embedding space
        PCA compression of samples
        K-Means clustering
        Select K samples (closest to each cluster mean)
    '''
    def __init__(self, seed_num:int=1):
        super().__init__(seed_num)
        model_path = '/home/alta/Conversational/OET/al826/2022/shortcuts/data_pruning/trained_models/boolq/baseline/0'   
        self.load_model(model_path)
     
    def filter_data(self, data:List, ret_frac:float, ncomps:int=10)->List:
        N = int(ret_frac*len(data))
        
        # Encoder embedding space
        H = self.get_hidden_vecs(data)

        # PCA compression
        pca = PCA(n_components=ncomps)
        compressed = pca.fit_transform(H)

        # K means clustering
        kmeans = KMeans(n_clusters=N, random_state=0).fit(compressed)
        
        # Select closest to mean per cluster
        cluster_means = kmeans.cluster_centers_
        # cluster_labels = kmeans.cluster_labels_
        selected_inds = []
        for k in range(N):
            # mask = cluster_labels == k
            # mask_scaler = ((mask-1)*10000000)+1
            mean_vector = cluster_means[k]
            diff_l2 = np.linalg.norm(compressed - mean_vector, axis=1)
            # diff_l2 = np.linalg.norm(compressed - mean_vector, axis=1) * mask_scaler
            ind = np.argmin(diff_l2)
            selected_inds.append(ind)
        return [data[ind] for ind in selected_inds]
        

    def get_hidden_vecs(self, data):
        H = [self.model_output(ex).h[0] for ex in data]
        H = torch.FloatTensor(H)
        return H
    
