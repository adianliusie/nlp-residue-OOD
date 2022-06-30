import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

class KMeansSelector():
    '''
        View samples in encoder embedding space
        PCA compression of samples
        K-Means clustering
        Select K samples (closest to each cluster mean)
    '''
    @classmethod
    def select(cls, input_ids, mask, model, device, bs=8, ncomps=10, frac_samples=0.1):

        # Encoder embedding space
        embeddings = cls._get_encoder_embeddings(input_ids, mask, model, device, bs).detach().cpu().numpy()

        # PCA compression
        pca = PCA(n_components=ncomps)
        compressed = pca.fit_transform(embeddings)

        # K means clustering
        num_samples = int(frac_samples*embeddings.size(0))
        kmeans = KMeans(n_clusters=num_samples, random_state=0).fit(compressed)
        
        # Select closest to mean per cluster
        cluster_means = kmeans.cluster_centers_
        cluster_labels = kmeans.cluster_labels_
        selected_samples = []
        for k in num_samples:
            valid_samples = compressed[cluster_labels==k]
            valid_mean = cluster_means[k]
            diff_l2 = np.linalg.norm(valid_samples - valid_mean, axis=1)
            ind = np.argmin(diff_l2)
            selected_samples.append(valid_samples[ind])
        return np.stack(selected_samples)



    @staticmethod
    def _get_encoder_embeddings(input_ids, mask, model, device, bs=8):
        '''
        Input is a tensor of input ids and mask
        Returns tensor of CLS embeddings at the correct layer
        Does this in batches
        '''
        emb = []
        ds = TensorDataset(input_ids, mask)
        dl = DataLoader(ds, batch_size=bs)
        with torch.no_grad():
            for id, m in dl:
                id = id.to(device)
                m = m.to(device)
                trans_output = model.transformer(id, m)
                H = trans_output.last_hidden_state  #[bsz, L, 768] 
                h = H[:, 0]                         #[bsz, 768] 
                emb.append(h.cpu())
        embeddings = torch.cat(h)
        return embeddings