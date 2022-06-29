import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class KMeansSelector():
    '''
        View samples in encoder embedding space
        PCA compression of samples
        K-Means clustering
        Select K samples (closest to each cluster mean)
    '''
    @staticmethod
    def select(model, ):
        pass

    def _get_encoder_embeddings(input_ids, mask, model, device, bs=8):
        '''
        Input is a tensor of input ids and mask
        Returns tensor of CLS embeddings at the correct layer
        Does this in batches
        '''
        CLS = []
        ds = TensorDataset(input_ids, mask)
        dl = DataLoader(ds, batch_size=bs)
        with torch.no_grad():
            for id, m in dl:
                id = id.to(device)
                m = m.to(device)
                layer_embeddings = model(id, m)
                CLS_embeddings = layer_embeddings[:,0,:].squeeze(dim=1)
                CLS.append(CLS_embeddings.cpu())
        embeddings = torch.cat(CLS)
        return embeddings