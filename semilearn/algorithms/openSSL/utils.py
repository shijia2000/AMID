import os
import numpy as np
import torch
import torch.nn.functional as F
from faiss import Kmeans as faiss_Kmeans
import networkx as nx
from sklearn.metrics import silhouette_score


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def t_sne(data,label,n_cluster, save_dir=None):
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data)
    x = data_tsne[:, 0]
    y = data_tsne[:, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, n_cluster))

    plt.clf()
    plt.figure(figsize=(10, 8))

    for i in range(n_cluster):
        plt.scatter(x[label==i],y[label==i],label=f'Cluster {i}',c=colors[i],alpha=0.5)


    plt.title(f't-SNE Visualization of Clustering Results, num_clusters={n_cluster}')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"cluster_{n_cluster}.png"))
    plt.show()

def pca_plt(data,label,n_cluster, save_dir=None):
    data_pca = preprocess_features(data)

    x = data_pca[:, 0]
    y = data_pca[:, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, n_cluster))

    plt.clf()
    plt.figure(figsize=(10, 8))

    for i in range(n_cluster):
        plt.scatter(x[label==i],y[label==i],label=f'Cluster {i}',c=colors[i],alpha=0.2)

    plt.title(f'PCA Visualization of Clustering Results, num_clusters={n_cluster}')
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"cluster_{n_cluster}.png"))
    plt.show()


def get_domain_cluster_adaptive(args, dataloader, print_fn=print):
    cluster_feat_path = os.path.join(args.cluster_dir, 'cluster_feat.npy')
    os.makedirs(args.cluster_dir, exist_ok=True)
    if hasattr(dataloader.dataset, 'cls_token_list'):
        features = dataloader.dataset.cls_token_list
        features = np.array(features)
        # features = torch.from_numpy(features)
        np.save(cluster_feat_path, features)
    else:
        raise ValueError('dataset do not have cls_token_list attribute')

    domain_label = dataloader.dataset.domains
    domain_label = np.array(domain_label)
    t_sne(features, domain_label, 4, './')

    # 设置min max num_clusters
    min_clusters = 2
    max_clusters = 10

    # 设置最佳聚类数量
    best_score = -1
    best_clusters = min_clusters

    tot_cluster_labels = []
    for num_clusters in range(min_clusters, max_clusters + 1):
        # _, cluster_label = single_kmeans(num_clusters, features, use_gpu=True)
        # cluster_label = cluster_label.cpu().detach().numpy()
        # tot_cluster_labels.append(cluster_label)

        cluster_method = CLUSTER_METHOD[args.cluster_method](num_clusters)
        cluster_method.cluster(features)
        cluster_label = arrange_clustering(cluster_method.images_lists)
        tot_cluster_labels.append(cluster_label)

        cluster_label_file = os.path.join(args.cluster_dir, f'cluster_label_cluster{num_clusters}.npy')
        np.save(cluster_label_file, cluster_label)


        score = silhouette_score(features, cluster_label)
        print_fn(f'For {num_clusters} clusters, silhouette score: {score}')
        if score > best_score:
            best_score = score
            best_clusters = num_clusters

        plt_dir = os.path.join(args.cluster_dir, 'plt_result')
        os.makedirs(plt_dir,exist_ok=True)

        t_sne(features,cluster_label,num_clusters, plt_dir)
        # pca_plt(features,cluster_label,num_clusters, plt_dir)


    args.num_domains = best_clusters
    print_fn(f'Best number of clusters:{best_clusters}')
    cluster_label = tot_cluster_labels[best_clusters - min_clusters]
    # cluster_label_path = os.path.join(args.cluster_dir, 'cluster_label.bin')
    # cluster_label.tofile(cluster_label_path)
    return best_clusters, cluster_label





def single_kmeans(num_centroids, data, init_centroids=None, frozen_centroids=False, seed=0, use_gpu=False):
    data = data.cpu().detach().numpy()
    feat_dim = data.shape[-1]
    if init_centroids is not None:
        init_centroids = init_centroids.cpu().detach().numpy()
    km = faiss_Kmeans(
        feat_dim,
        num_centroids,
        niter=40,
        verbose=False,
        spherical=True,
        min_points_per_centroid=int(data.shape[0] / num_centroids * 0.9),
        gpu=use_gpu,
        seed=seed,
        frozen_centroids=frozen_centroids
    )
    km.train(data, init_centroids=init_centroids)
    _, assignments = km.index.search(data, 1)
    centroids = torch.from_numpy(km.centroids).cuda()
    assignments = torch.from_numpy(assignments).long().cuda().squeeze(1)

    return centroids, assignments


def multi_kmeans(K_list, data, seed=0, init_centroids=None, frozen_centroids=False):
    if init_centroids is None:
        init_centroids = [None] * len(K_list)
    centroids_list = []
    assignments_list = []
    for idx, K in enumerate(K_list):
        centroids, assignments = single_kmeans(K,
                                               data,
                                               init_centroids=init_centroids[idx],
                                               frozen_centroids=frozen_centroids,
                                               seed=seed+idx)
        centroids_list.append(centroids)
        assignments_list.append(assignments)

    return centroids_list, assignments_list


def update_semantic_prototypes(feats_lb, labels_lb, num_classes):
    feats = feats_lb
    labels = labels_lb
    feat_dim = feats.shape[1]
    prototypes = torch.zeros((num_classes, feat_dim)).cuda() + 1e-7
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = F.normalize(feats[mask].mean(0), dim=0)
    return prototypes


def update_fused_semantic_prototypes(feats_lb, labels_lb, feats_ulb, plabels_ulb, num_classes):
    feats = torch.cat((feats_lb, feats_ulb), dim=0)
    labels = torch.cat((labels_lb, plabels_ulb))
    feat_dim = feats.shape[1]
    prototypes = torch.zeros((num_classes, feat_dim)).cuda() + 1e-7
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = F.normalize(feats[mask].mean(0), dim=0)
    return prototypes


def get_cluster_labels(semantic_prototypes, structural_centroids, min_cluster=0.75):
    num_classes = semantic_prototypes.shape[0]
    num_clusters = structural_centroids.shape[0]
    cluster_scale = int(min_cluster * num_clusters / num_classes)
    dists = torch.cdist(structural_centroids, semantic_prototypes)
    G = nx.DiGraph()
    # Here with the networkx package, the demand value is actually "-b(v)" presented in the paper
    # Therefore, positive demand value means it is a demand node, and negative for a supply node
    for u in range(num_clusters):
        G.add_node(f"st_{u}", demand=-1)
        for v in range(num_classes):
            if u == 0:
                G.add_node(f"se_{v}", demand=cluster_scale)
            G.add_edge(f"st_{u}", f"se_{v}", capacity=1, weight=int(dists[u, v] * 1000))
            if v == 0:
                G.add_node(f"sink", demand=num_clusters - num_classes * cluster_scale)
            if u == 0:
                G.add_edge(f"se_{v}", "sink")
    flow = nx.min_cost_flow(G)
    cluster_labels = torch.zeros(num_clusters).long().cuda()
    for u in range(num_clusters):
        for v in range(num_classes):
            if flow[f"st_{u}"][f"se_{v}"] > 0:
                cluster_labels[u] = v
    return cluster_labels


def get_cluster_labels_simple(semantic_prototypes, structural_centroids, T):
    sim = torch.exp(torch.mm(structural_centroids, semantic_prototypes.t()) / T)
    sim_probs = sim / sim.sum(1, keepdim=True)
    _, cluster_labels = sim_probs.max(1)
    return cluster_labels


def get_shadow_centroids(structural_assignments, num_centroids, feats):
    feat_dim = feats.shape[1]
    shadow_centroids = torch.zeros((num_centroids, feat_dim)).cuda()
    for c in range(num_centroids):
        mask = (structural_assignments == c)
        shadow_centroids[c] = feats[mask].mean(0)
    return shadow_centroids


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class Clustering:
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        self.k = k
        self.pca_dim = pca_dim
        self.whitening = whitening
        self.L2norm = L2norm

    def cluster(self, data):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        # PCA-reducing, whitening and L2-normalization
        self.xb = preprocess_features(data, self.pca_dim, self.whitening, self.L2norm)
        # cluster the data
        I = self.run_method(self.xb, self.k)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)


    def run_method(self):
        print('Define each method')

def preprocess_features(npdata, pca_dim=256, whitening=False, L2norm=False):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca_dim (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')
    pca = PCA(pca_dim, whiten=whitening)
    npdata = pca.fit_transform(npdata)
    # L2 normalization
    if L2norm:
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]
    return npdata


class Kmeans(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        I = kmeans.fit_predict(x)
        return I

class GMM(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters):
        gmm = GaussianMixture(n_components=n_clusters)
        I = gmm.fit_predict(x)
        return I

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]

CLUSTER_METHOD={
    'Kmeans':Kmeans,
    'GMM':GMM,
    # 'Spectral':Spectral,
    # 'Agglomerative':Agglomerative,
    # 'DPC':DPC
}


"""
softmatch about
"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


class SoftMatchWeightingHook(MaskingHook):
    """
    SoftMatch learnable truncated Gaussian weighting
    """

    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)
        else:
            self.prob_max_mu_t = torch.ones((self.num_classes)) / self.args.num_classes
            self.prob_max_var_t = torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = self.concat_all_gather(probs_x_ulb)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs)  # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return max_probs, max_idx

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask

"""
softmatch about end
"""



