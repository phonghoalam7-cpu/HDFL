import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np
import os


def get_office_dataset(args):
    """
    加载 Office-Caltech10 数据集，并按域（Domain）分配给客户端。
    支持的域: Amazon, Caltech, DSLR, Webcam
    """
    data_dir = './data/office_caltech_10/'
    domains = ['amazon', 'caltech', 'dslr', 'webcam']

    # 检查数据是否存在
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Office-Caltech10 dataset not found at {data_dir}. Please download it.")

    # 1. 定义数据预处理 (ResNet 标准输入: 224x224)
    # 训练集做增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 测试集只做 Resize 和 CenterCrop
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 分别加载每个域的数据
    domain_train_datasets = []
    domain_test_datasets = []

    # 用于记录每个域在 ConcatDataset 中的索引范围 [start, end)
    domain_train_ranges = {}
    domain_test_ranges = {}

    current_train_offset = 0
    current_test_offset = 0

    print(f'Loading Office-Caltech10 domains: {domains} ...')

    for domain in domains:
        domain_path = os.path.join(data_dir, domain)
        if not os.path.exists(domain_path):
            raise RuntimeError(f"Domain folder not found: {domain_path}")

        # 加载完整数据集用于获取长度和划分索引
        full_dataset = datasets.ImageFolder(root=domain_path)
        num_samples = len(full_dataset)
        indices = np.arange(num_samples)

        # 设定随机种子保证可复现性，否则每次运行划分都不一样
        np.random.seed(args.seed)
        np.random.shuffle(indices)

        # 划分 Train/Test (80% 训练, 20% 测试)
        split = int(0.8 * num_samples)
        train_idxs, test_idxs = indices[:split], indices[split:]

        # 重新加载两次以应用不同的 transform
        # 注意：ImageFolder 加载很快，因为它只是扫描文件名，图片是惰性加载的
        train_ds_raw = datasets.ImageFolder(root=domain_path, transform=train_transform)
        test_ds_raw = datasets.ImageFolder(root=domain_path, transform=test_transform)

        # 创建 Subset
        train_subset = Subset(train_ds_raw, train_idxs)
        test_subset = Subset(test_ds_raw, test_idxs)

        domain_train_datasets.append(train_subset)
        domain_test_datasets.append(test_subset)

        # --- 记录训练集索引范围 ---
        len_train = len(train_idxs)
        domain_train_ranges[domain] = (current_train_offset, current_train_offset + len_train)
        current_train_offset += len_train

        # --- 记录测试集索引范围 ---
        len_test = len(test_idxs)
        domain_test_ranges[domain] = (current_test_offset, current_test_offset + len_test)
        current_test_offset += len_test

    # 3. 合并所有域的数据
    # global_train_dataset 索引范围: 0 ~ sum(len_train)
    # global_test_dataset 索引范围: 0 ~ sum(len_test)
    global_train_dataset = ConcatDataset(domain_train_datasets)
    global_test_dataset = ConcatDataset(domain_test_datasets)

    # 4. 构建 user_groups (分配逻辑)
    # 目标：每个 User 分配到一个特定的 Domain
    user_groups = {}  # 训练索引
    test_user_groups = {}  # 测试索引

    num_users = args.num_users

    for client_idx in range(num_users):
        # 轮询选择域
        domain_name = domains[client_idx % len(domains)]

        # --- 处理训练集分配 ---
        tr_start, tr_end = domain_train_ranges[domain_name]
        train_available_indices = np.arange(tr_start, tr_end)

        # --- 处理测试集分配 ---
        te_start, te_end = domain_test_ranges[domain_name]
        test_available_indices = np.arange(te_start, te_end)

        # 如果多个客户端共享同一个域，切分数据
        sharers = [i for i in range(num_users) if domains[i % len(domains)] == domain_name]
        sharer_rank = sharers.index(client_idx)

        # 切分训练集
        client_train_indices = np.array_split(train_available_indices, len(sharers))[sharer_rank]
        user_groups[client_idx] = client_train_indices.tolist()

        # 切分测试集 (即使只有一个客户端，tolist() 也是必须的)
        client_test_indices = np.array_split(test_available_indices, len(sharers))[sharer_rank]
        test_user_groups[client_idx] = client_test_indices.tolist()

    return global_train_dataset, global_test_dataset, user_groups, test_user_groups