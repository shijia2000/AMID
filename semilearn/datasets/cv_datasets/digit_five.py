import os
import numpy as np

from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data, find_classes, make_dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



domain_list = ['svhn', 'mnist', 'mnist_m', 'usps', 'synth']

def get_digit_five(args, alg, num_labels, data_dir='./data'):
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    num_classes = 10
    labeled_domain = args.labeled_domain
    unlabeled_domain = args.unlabeled_domain

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dirs = {}
    test_dirs = {}
    for domain in domain_list:
        train_dirs[domain] = os.path.join(data_dir, 'digit_five', domain, 'train')
        test_dirs[domain] = os.path.join(data_dir, 'digit_five', domain, 'test')

    classes, class_to_idx = find_classes(train_dirs['svhn'])

    train_data_dict = {}
    train_targets_dict = {}
    test_data_dict = {}
    test_targets_dict = {}
    for domain in domain_list:
        train_data_dict[domain], train_targets_dict[domain] = make_dataset(train_dirs[domain], class_to_idx)
        test_data_dict[domain], test_targets_dict[domain] = make_dataset(test_dirs[domain], class_to_idx)

    lb_data_dict = {}
    lb_targets_dict = {}
    ulb_data_dict = {}
    ulb_targets_dict = {}

    # 域标签
    lb_domain_dic = {}
    ulb_domain_dic = {}
    domain_idx = 0
    for domain in domain_list:
        lb_data_dict[domain], lb_targets_dict[domain], \
            ulb_data_dict[domain], ulb_targets_dict[domain] = split_ssl_data(args,
                                                                             train_data_dict[domain],
                                                                             train_targets_dict[domain],
                                                                             num_classes=num_classes,
                                                                             lb_num_labels=num_labels,
                                                                             domain=domain)

        lb_domain_dic[domain] = [domain_idx] * len(lb_data_dict[domain])
        ulb_domain_dic[domain] = [domain_idx] * len(ulb_data_dict[domain])
        domain_idx = domain_idx + 1

    lb_data, lb_targets = lb_data_dict[labeled_domain], lb_targets_dict[labeled_domain]
    in_eval_data, in_eval_targets = test_data_dict[labeled_domain], test_targets_dict[labeled_domain]

    ulb_data = np.concatenate([ulb_data_dict[domain] for domain in unlabeled_domain])
    ulb_targets = np.concatenate([ulb_targets_dict[domain] for domain in unlabeled_domain])

    out_eval_data = np.concatenate([test_data_dict[domain] for domain in unlabeled_domain if domain != labeled_domain])
    out_eval_targets = np.concatenate([test_targets_dict[domain] for domain in unlabeled_domain if domain != labeled_domain])

    all_eval_data = np.concatenate([in_eval_data, out_eval_data])
    all_eval_targets = np.concatenate([in_eval_targets, out_eval_targets])

    from .myDataset import myBasicDataset
    lb_dset = myBasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)
    ulb_dset = myBasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)
    eval_dset = myBasicDataset(alg, in_eval_data, in_eval_targets, num_classes, transform_val, False, None, False)
    out_dset = myBasicDataset(alg, out_eval_data, out_eval_targets, num_classes, transform_val, False, None, False)
    all_eval_dset = myBasicDataset(alg, all_eval_data, all_eval_targets, num_classes, transform_val, False, None, False)

    # get unlabled data's cls_token
    from .cls_token import cal_cls_token
    u_cls_token_list = cal_cls_token(args, ulb_dset)
    ulb_dset.set_cls_token(u_cls_token_list)

    lb_domain_labels = lb_domain_dic[labeled_domain]
    ulb_domain_labels = np.concatenate([ulb_domain_dic[domain] for domain in domain_list])
    lb_dset.set_domain(lb_domain_labels)
    ulb_dset.set_domain(ulb_domain_labels)

    return lb_dset, ulb_dset, eval_dset, out_dset, all_eval_dset


"""
load from mat
"""



