from .model_analyser import PcaAnalyser

class ResidueDetector(PcaAnalyser):
    '''
    Usage pipeline:
        1) self.train(args for data to train on for PCA)
        2) self.make_gaussian(args for data to define Gaussian in PCA feature space)
        2) in_scores = self.gaussian_nll_score(args for in domain data)
        3) out_scores = self.gaussian_nll_score(args for ood data)
        4) calc_f1(in_scores, out_score)
    '''
    def gaussian_nll_score(self, data_name:str, mode:str, lim:int=None, quiet=False):
        pass
