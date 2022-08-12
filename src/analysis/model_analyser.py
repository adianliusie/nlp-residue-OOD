import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from transformers import BertModel
from tqdm import tqdm
from sklearn.decomposition import PCA
from functools import lru_cache
from types import SimpleNamespace

from ..system_loader import SystemLoader
from ..utils.torch_utils import no_grad
from ..helpers import DirHelper

class ModelAnalyser(SystemLoader):
    def __init__(self, exp_path:str):
        self.dir = DirHelper.load_dir(exp_path)
        self.set_up_helpers()
        
    def reset_model(self):
        self.model  = BertModel.from_pretrained("bert-base-uncased")
    
    @lru_cache(maxsize = 10)
    @no_grad
    def get_cls_vectors(self, data_name:str, mode:str, lim:int=None, quiet=False)->dict:
        """get model predictions for given data"""
        self.model.eval()
        self.to(self.device)
        eval_data = self.data_loader.get_data_split(data_name, mode, lim)
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)
        
        CLS_vectors = {}
        for batch in tqdm(eval_batches, disable=quiet):
            sample_id = batch.sample_id[0]
            output = self.model_output(batch)

            h = output.h.squeeze(0)
            CLS_vectors[sample_id] = h.cpu().numpy()
        return CLS_vectors
    
    @lru_cache(maxsize = 10)
    @no_grad
    def get_probs(self, data_name:str, mode:str, lim:int=None, quiet=False)->dict:
        self.model.eval()
        self.to(self.device)
        eval_data = self.data_loader.get_data_split(data_name, mode, lim)
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)
        
        output_probs = {}
        for batch in tqdm(eval_batches, disable=quiet):
            sample_id = batch.sample_id[0]
            output = self.model_output(batch)

            y = output.y
            prob = torch.softmax(y, dim=-1).squeeze(0)
            output_probs[sample_id] = prob.cpu().numpy()
            
        return output_probs        
    
    def get_hits(self, data_name:str, mode:str)->dict:
        preds = self.load_preds(data_name, mode='test')
        labels = self.load_labels(data_name, mode='test')
        assert preds.keys() == labels.keys()
        return {k: preds[k] == labels[k] for k in labels.keys()}
    
    def model_output(self, batch):
        """ overwrite method to not break with different labels """
        output = self.model(input_ids=batch.ids, attention_mask=batch.mask)
        
        preds = torch.argmax(output.y, dim=-1)
        if set(batch.labels.tolist()).issubset(preds.tolist()): 
            loss = F.cross_entropy(output.y, batch.labels)
        else:
            loss = 0
            
        # return accuracy metrics
        hits = torch.argmax(output.y, dim=-1) == batch.labels
        hits = torch.sum(hits[batch.labels != -100]).item()
        num_preds = torch.sum(batch.labels != -100).item()
        return SimpleNamespace(loss=loss, y=output.y, h=output.h,
                               hits=hits, num_preds=num_preds)

class PcaAnalyser(ModelAnalyser):
    def train_pca(self, data_name:str, mode:str, lim:int=None, quiet=False):
        self.pca = PCA()
        cls_vectors = self.get_cls_vectors(data_name, mode, lim, quiet)
        matrix = [i for i in cls_vectors.values()]
        self.pca.fit(matrix)
        
    def run_pca(self, data_name:str, mode:str, lim:int=None, quiet=False):
        cls_vectors = self.get_cls_vectors(data_name, mode, lim, quiet)
        matrix = [i for i in cls_vectors.values()]
        output = self.pca.transform(matrix)
        return output
        
    def plot_components(self, data_name:str, mode:str, lim:int=None, quiet=False):
        matrix = self.run_pca(data_name, mode, lim, quiet)
        
        #get mean and 2 std lines
        abs_matrix = np.abs(matrix)
        mean = np.mean(abs_matrix, axis=0)
        lower = np.percentile(abs_matrix, 20, axis=0)
        upper = np.percentile(abs_matrix, 80, axis=0)

        #plot components
        x = range(matrix.shape[1])
        plt.plot(x, mean)
        plt.fill_between(x, lower, upper, alpha=0.2)
        

        