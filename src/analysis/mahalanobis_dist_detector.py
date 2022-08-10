import numpy as np
from .model_analyser import ModelAnalyser

class MahalDistDetector(ModelAnalyser):
    '''
    Implementation of: https://arxiv.org/pdf/1807.03888.pdf
    '''
    def __init__(self, exp_path:str):
        super().__init__(exp_path)
    
    def train_detector(self, data_name:str, mode:str, lim:int=None, quiet=False):
        cls_vectors = self.get_cls_vectors(data_name, mode, lim, quiet)
        labels = self.load_labels(data_name, mode=mode)
        num_classes = len(set(labels.values()))

        class_means = []
        cov = np.zeros((cls_vectors.shape[-1], cls_vectors.shape[-1]))
        for c in range(num_classes):
            matrix = np.array([cls_vectors[id] for id in labels.keys() if labels[id]==c])
            class_means.append(np.mean(matrix, axis=0))
            cov += np.cov(matrix, rowvar=False)
        cov = cov/num_classes
        inv_cov = np.linalg.inv(cov)

        self.class_means = class_means
        self.inv_cov = inv_cov
    
    def calc_mahal_dists(self, data_name:str, mode:str, lim:int=None, quiet=False):
        cls_vectors = self.get_cls_vectors(data_name, mode, lim, quiet)
        mahal_dists = {}
        for id, vec in cls_vectors.items():
            mahal_dists[id] = self.mahal_dist(vec, self.class_means, self.inv_cov)
        return mahal_dists
    
    @classmethod
    def mahal_dist(cls, vector, class_means, inv_cov):
        dists = []
        for class_mean in class_means:
            dists.append(cls.calculate_per_class_dist(vector, class_mean, inv_cov))
        return min(dists)
    
    @staticmethod
    def calculate_per_class_dist(vector, class_mean, inv_cov):
        diff = vector - class_mean
        half = np.matmul(inv_cov, diff)
        return np.dot(diff, half)