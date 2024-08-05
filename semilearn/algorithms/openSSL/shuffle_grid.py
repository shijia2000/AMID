import copy
import os
import random
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import torch.utils.data as data


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from .utils import CLUSTER_METHOD

im_size = 32
grid =3
batch_size= 512
epochs= 300
temperature= 0.5
use_cosine_similarity= True
out_dim= 64
weight_decay= 10e-6
log_every_n_steps= 50

def load_img(img_path, im_size, grid):
    img = Image.open(img_path).convert('RGB')

    s = int(im_size / grid)
    tile = [
        img.crop(np.array([s * (n % grid), s * int(n / grid), s * (n % grid + 1), s * (int(n / grid) + 1)]).astype(int))
        for n in range(grid ** 2)]
    random.shuffle(tile)
    dst = Image.new('RGB', (int(s * grid), int(s * grid)))
    for i, t in enumerate(tile):
        dst.paste(t, (i % grid * s, int(i / grid) * s))
    img = dst
    return img

class conDataset(data.Dataset):
    def __init__(self, data, labels, transform=None, im_size=32, grid = 3):
        self.data=data
        self.labels=labels
        self.transform=transform
        self.im_size=im_size
        self.grid=grid

    def __getitem__(self, index):
        img_path = self.data[index]

        img = load_img(img_path,self.im_size,self.grid)
        img_1 = self.transform(img)
        img_2 = self.transform(img)
        return img_1, img_2

    def __len__(self):
        return len(self.data)


def train_domain_split(args, dataset):
    dataset = copy.deepcopy(dataset)
    data = dataset.data
    labels = dataset.targets
    domain_labels = dataset.domains

    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=im_size, scale=(0.08, 1.0)),
                                          transforms.Grayscale(3),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                          ])

    dataset_grid = conDataset(data,labels,data_transforms,im_size,grid)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = torch.utils.data.DataLoader(dataset_grid, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    nt_xent_criterion = NTXentLoss(device,batch_size,temperature,use_cosine_similarity)

    model = Encoder(3, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=weight_decay)

    n_iter = 0
    valid_n_iter = 0
    best_valid_loss = np.inf

    model.train()

    model_checkpoints_folder = os.path.join('grid_checkpoints')
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    model_path = os.path.join(model_checkpoints_folder, 'model.pth')

    # for epoch_counter in range(epochs):
    #     for xis, xjs in train_loader:
    #         optimizer.zero_grad()
    #
    #         xis = xis.to(device)
    #         xjs = xjs.to(device)
    #
    #         # loss = _step(model, xis, xjs, n_iter)
    #         # get the representations and the projections
    #         ris, zis = model(xis)  # [N,C]
    #
    #         # get the representations and the projections
    #         rjs, zjs = model(xjs)  # [N,C]
    #
    #         # normalize projection feature vectors
    #         zis = F.normalize(zis, dim=1)
    #         zjs = F.normalize(zjs, dim=1)
    #
    #         loss = nt_xent_criterion(zis, zjs)
    #
    #         if n_iter % log_every_n_steps == 0:
    #             print(f'Epoch:{epoch_counter}/{epochs}({n_iter}) loss:{loss}')
    #
    #         # optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         n_iter += 1
    #
    #     # model_checkpoints_folder = os.path.join('grid_checkpoints')
    #     # os.makedirs(model_checkpoints_folder,exist_ok=True)
    #     # model_path = os.path.join(model_checkpoints_folder, 'model.pth')
    #     torch.save(model.state_dict(), model_path)

    model.eval()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    dataloader = torch.utils.data.DataLoader(dataset_grid, batch_size=56, shuffle=False)

    feats = []
    with torch.no_grad():
        for im, _ in dataloader:
            im = im.cuda()
            feat, _ = model(im)

            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats)

    plt_dir = os.path.join(model_checkpoints_folder, 'grid_result')
    os.makedirs(plt_dir, exist_ok=True)

    # tsne plot
    # tsne = TSNE(n_components=2, perplexity=30, verbose=1, n_jobs=3)
    tsne = TSNE(n_components=2, random_state=0)
    feats_decomposit = tsne.fit_transform(feats)

    # plt.clf()
    # plt.figure(figsize=(10, 8))
    # plt.scatter(feats_decomposit[:, 0], feats_decomposit[:, 1], c=domain_labels, alpha=0.5)
    # plt.savefig(os.path.join(plt_dir, f"tsne_feats.png"))
    path = os.path.join(plt_dir, f"tsne_feats.png")
    my_plt(feats_decomposit, domain_labels,len(np.unique(domain_labels)), path)

    pca = PCA()
    feats_pca = pca.fit_transform(feats)
    # plt.clf()
    # plt.figure(figsize=(10, 8))
    # plt.scatter(feats_pca[:, 0], feats_pca[:, 1], c=domain_labels,alpha=0.5)
    # plt.savefig(os.path.join(plt_dir, f"pca_feats.png"))
    path = os.path.join(plt_dir, f"pca_feats.png")
    my_plt(feats_pca, domain_labels, len(np.unique(domain_labels)), path)


    cluster_dir_kmeans = os.path.join(plt_dir, 'cluster_res', 'kmeans')
    cluster_dir_gmm = os.path.join(plt_dir, 'cluster_res', 'gmm')
    os.makedirs(cluster_dir_kmeans)
    os.makedirs(cluster_dir_gmm)


    # 设置min max num_clusters
    min_clusters = 2
    max_clusters = 10

    # 设置最佳聚类数量
    best_score = -1
    best_clusters = min_clusters

    tot_cluster_labels_kMeans = []
    tot_cluster_labels_GMM = []
    for num_clusters in range(min_clusters, max_clusters + 1):
        # kmeans = CLUSTER_METHOD['Kmeans'](num_clusters)
        # gmm = CLUSTER_METHOD['GMM'](num_clusters)

        kmeans = KMeans(num_clusters)
        kmeans_cluster = np.array(kmeans.fit_predict(feats_pca))
        tot_cluster_labels_kMeans.append(kmeans_cluster)

        gmm = GMM(num_clusters)
        gmm_cluster = np.array(gmm.fit_predict(feats_pca))
        tot_cluster_labels_GMM.append(gmm_cluster)

        # cluster_label_file = os.path.join(args.cluster_dir, f'cluster_label_cluster{num_clusters}.npy')
        # np.save(cluster_label_file, cluster_label)

        score_kmeans = silhouette_score(feats_pca, kmeans_cluster)
        score_gmm = silhouette_score(feats_pca, gmm_cluster)
        print(f'For {num_clusters} clusters, silhouette score: {score_gmm}')
        if score_gmm > best_score:
            best_score = score_gmm
            best_clusters = num_clusters

        # plt.clf()
        # plt.scatter(feats_pca[:, 0], feats_pca[:, 1], c=gmm_cluster, alpha=0.5)
        # plt.savefig(os.path.join(plt_dir, f"pca_feats_{num_clusters}.png"))

        path = os.path.join(plt_dir, f"pca_gmm_cluster_{num_clusters}.png")
        my_plt(feats_pca, gmm_cluster, len(np.unique(gmm_cluster)), path)

    # t_sne(features, cluster_label, num_clusters, plt_dir)
    print(f'Best number of clusters:{best_clusters}')


def my_plt(feat, label, n_cluster, save_dir=None):
    x = feat[:, 0]
    y = feat[:, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, n_cluster))
    plt.clf()
    plt.figure(figsize=(10, 8))

    for i in range(n_cluster):
        plt.scatter(x[label == i], y[label == i], label=f'Cluster {i}', c=colors[i], alpha=0.5)

    plt.title(f't-SNE Visualization, num_clusters={n_cluster}')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend()
    # plt.savefig(os.path.join(save_dir, f"cluster_{n_cluster}.png"))
    plt.savefig(save_dir)
    plt.show()

import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, input_dim=1, out_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # projection MLP
        self.l1 = nn.Linear(64, 64)
        self.fc_bn1 = nn.BatchNorm1d(64)
        self.l2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, 2, 2)

        h = torch.mean(x, dim=[2, 3])

        x = self.l1(h)
        x = F.relu(x)
        x = self.fc_bn1(x)
        x = self.l2(x)

        return h, x