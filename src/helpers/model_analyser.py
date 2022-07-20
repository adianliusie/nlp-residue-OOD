from transformers import BertModel

from ..system_loader import SystemLoader

class ModelAnalyser(SystemLoader):
    def __init__(self, exp_path:str):
        self.dir = DirHelper.load_dir(exp_path)
        self.set_up_helpers()
        
    def reset_model(self):
        self.model  = BertModel.from_pretrained("bert-base-uncased")
    
    @no_grad
    def get_cls_vectors(self, data_name:str, mode:str)->dict:
        """get model predictions for given data"""
        self.model.eval()
        self.to(self.device)
        eval_data = self.data_loader.get_data_split(data_name, mode)
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)
        
        CLS_vectors = {}
        for batch in tqdm(eval_batches):
            sample_id = batch.sample_id[0]
            output = self.model_output(batch)

            h = output.h.squeeze(0)
            CLS_vectors[sample_id] = h.cpu().numpy()
        return CLS_vectors
    
    def get_hits(self, data_name:str, mode:str)->dict:
        preds = self.load_preds(data_name:str, mode='test')
        labels = self.load_labels(data_name:str, mode='test')
        assert preds.keys() == labels.keys()
        return {k: preds[k] == labels[k] for k in labels.keys()}