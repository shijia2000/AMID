# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import get_onehot


class myBasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args,
                 **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(myBasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel',
                                        'meanteacher', 'mixmatch', 'cafa'], \
                    f"alg {self.alg} requires strong augmentation"

        # self.clusters = None
        # self.cls_token_list = None
        self.get_cluster = False
        self.get_domain = False
        for arg, var in kwargs.items():
            setattr(self, arg, var)


    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        img = self.data[idx]
        return img, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.__sample__(idx)

        if self.transform is None:
            return {'x_lb': transforms.ToTensor()(img), 'y_lb': target}
        else:
            # print(img)
            # if isinstance(img, np.ndarray):
            #     img = Image.fromarray(img)
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert('RGB')
            img_w = self.transform(img)
            # print(img_w.shape)
            if not self.is_ulb:
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target}
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w}
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1': img_s2,
                            'x_ulb_s_0_rot': img_s1_rot, 'rot_v': rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img),
                            'x_ulb_s_1': self.strong_transform(img)}
                else:
                    if hasattr(self, 'get_cluster') and hasattr(self, 'clusters'):
                        cluster = np.copy(self.clusters[idx])
                        cluster = np.array(cluster)
                        # cluster = torch.from_numpy(cluster)
                        return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img), 'y_ulb': target, 'cluster_ulb': cluster}

                    else:
                        return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img), 'y_ulb': target}



    def __len__(self):
        return len(self.data)


    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.data):
            raise ValueError("The length of cluster_list must to be same as self.data")
        else:
            setattr(self, 'clusters', cluster_list)
            self.get_cluster = True
            # self.clusters = cluster_list

    def set_domain(self, domain_list):
        if len(domain_list) != len(self.data):
            raise ValueError("The length of cluster_list must to be same as self.data")
        else:
            setattr(self, 'domains', domain_list)
            self.get_domain = True

    def set_cls_token(self, cls_token_list):
        if len(cls_token_list) != len(self.data):
            raise ValueError("The length of cls_token_list must to be same as self.data")
        else:
            # self.cls_token_list = cls_token_list
            setattr(self, 'cls_token_list', cls_token_list)




















