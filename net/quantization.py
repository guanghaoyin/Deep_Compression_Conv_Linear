import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

#随着bits的增加，识别效果更好
#但是当剪枝后权值矩阵中的非零个数少于2^bits时，无法进行聚类，需要减小bits以减小聚类类别个数
def apply_weight_sharing(model, bits=3):
    """
    Applies weight sharing to the given model
    """
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        # print(module)
        # print(len(shape))
        if len(shape) == 2:
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
        elif len(shape) == 4:
            mat = csr_matrix(weight.reshape(shape[0]*shape[2],shape[1]*shape[3])) if shape[0]*shape[2] < shape[1]*shape[3] else csc_matrix(weight.reshape(shape[0]*shape[2],shape[1]*shape[3]))
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2 ** bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                            precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            module.weight.data = torch.from_numpy(mat.toarray().reshape(shape[0], shape[1], shape[2], shape[3])).to(dev)

