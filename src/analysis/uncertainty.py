import numpy as np
import matplotlib.pyplot as plt 
from typing import Tuple
from sklearn.metrics import precision_recall_curve

from .model_analyser import PcaAnalyser
from ..utils.evaluation import FScore

class UncertaintyAnalyser(PcaAnalyser):
    def __init__(self, exp_path:str, method:str='max_prob'):
        super().__init__(exp_path)
        self.set_ood_approach(method)
        
    def ood_detection_pr(self, iid_name:str, ood_name:str, lim:int=None):
        iid_scores, ood_scores = self.ood_classification(iid_name, ood_name, lim)
        scores = iid_scores + ood_scores
        labels = [0]*len(iid_scores) + [1]*len(ood_scores)
        
        lr_precision, lr_recall, _ = precision_recall_curve(labels, scores)
        
        F1, P, R = max([(FScore(p, r), p, r) for p, r in zip(lr_precision, lr_recall)])
        print(f'{self.method} F1:{F1:.3f}   P:{P:.3f}   R:{R:.3f}') 
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic', linewidth=0.5)
    
    def set_ood_approach(self, method:str):
        """ method to set OOD detection approach """
        if   method == 'residue':     self.ood_classification = self.residue_detection
        elif method == 'max_prob':    self.ood_classification = self.max_prob_detection
        elif method == 'vec_dist':    self.ood_classification = self.vector_dist
        elif method == 'mahalanobis': self.ood_classification = self.mahalanobis
        else: raise ValueException('invalid OOD detection approach provided')
        self.method = method

    ### Various OOD detection approach methods ###########################################################
    
    def residue_detection(self, iid_name:str, ood_name:str, lim:int=None)->Tuple[list, list]:
        """ residue approach for OOD detection """
        #run PCA on train data, and calculate means and stds for each component
        self.train_pca(iid_name, 'train', lim=lim)
        train_vecs = self.run_pca(iid_name, 'train', lim=lim)
        mean = np.mean(train_vecs, axis=0) 
        var = np.var(train_vecs, axis=0) 
        iid_gaussian_score = lambda vec: sum([(vec[k]-mean[k])**2/(2*var[k]) for k in range(len(vec))])
        
        #look at in domain predictions
        iid_vecs   = self.run_pca(iid_name, 'test', lim=lim)
        iid_scores = [iid_gaussian_score(vec) for vec in iid_vecs]
        
        #look at out of domain predictions
        ood_vecs = self.run_pca(ood_name, 'test', lim=lim)
        ood_scores = [iid_gaussian_score(vec) for vec in ood_vecs]
        return iid_scores, ood_scores

    def vector_dist(self, iid_name:str, ood_name:str, lim:int=None)->Tuple[list, list]:
        train_vecs = self.get_cls_vectors(iid_name, 'train', lim=lim)
        train_vecs = [vec for vec in train_vecs.values()]
        mean = np.mean(train_vecs, axis=0) 
        var  = np.var(train_vecs, axis=0) 
        iid_gaussian_score = lambda vec: sum([(vec[k]-mean[k])**2/(2*var[k]) for k in range(len(vec))])

        #look at IID scores
        iid_vecs = self.get_cls_vectors(iid_name, 'test', lim=lim)
        iid_scores = [iid_gaussian_score(vec) for vec in iid_vecs.values()]
        
        #look at OOD scores
        ood_vecs = self.get_cls_vectors(ood_name, 'test', lim=lim)
        ood_scores = [iid_gaussian_score(vec) for vec in ood_vecs.values()]
        return iid_scores, ood_scores


    def max_prob_detection(self, iid_name:str, ood_name:str, lim:int=None)->Tuple[list, list]:
        """ maximum probability approach for OOD detection """
        #Get IID test probs
        iid_probs = self.get_probs(iid_name, 'test', lim=lim)
        iid_scores = [-1*max(prob) for prob in iid_probs.values()]

        #Get OOD test probs
        ood_probs = self.get_probs(ood_name, 'test', lim=lim)
        ood_scores = [-1*max(prob) for prob in ood_probs.values()]
        return iid_scores, ood_scores

    def mahalanobis(self, iid_name:str, ood_name:str, lim:int=None)->Tuple[list, list]:
        cls_vectors = self.get_cls_vectors(iid_name, 'train', lim)
        
        #get means of the different classes and the covariance matrix
        labels = self.load_labels(iid_name, 'train', lim) 
        classes = set(labels.values())
        
        class_means = []
        covariances = []
        for c in classes:
            class_vecs = np.array([cls_vectors[k] for k in labels.keys() if labels[k]==c])
            class_means.append(np.mean(class_vecs, axis=0))
            covariances.append(np.cov(class_vecs, rowvar=False))
        mean_cov = np.mean(covariances, axis=0)
        inv_cov = np.linalg.inv(mean_cov)

        #define mahalanbois distance funnction
        def mahalanobis_dist(vector):
            dists = []
            for class_mean in class_means:
                diff = vector - class_mean
                half = np.matmul(inv_cov, diff)
                dist = np.dot(diff, half)
                dists.append(dist)
            return min(dists)

        #Get IID test distance scores
        iid_vecs = self.get_cls_vectors(iid_name, 'test', lim=lim)
        iid_scores = [mahalanobis_dist(vec) for vec in iid_vecs.values()]

        #Get OOD test distance scores
        ood_vecs = self.get_cls_vectors(ood_name, 'test', lim=lim)
        ood_scores = [mahalanobis_dist(vec) for vec in ood_vecs.values()]
        return iid_scores, ood_scores
        
    ######################################################################################################