
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

class ComboIter(object):
    """迭代器：同时从多个 DataLoader 里取 batch"""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # 当最短的 DataLoader 结束时，停止
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

def safe_structure(batch):
    """
    强制保持 batch 结构: (data, label)，其中 data 至少是 list[list[tensor]]
    """
    if isinstance(batch, tuple) and len(batch) == 2:
        data, label = batch
    else:
        return batch  # 意外情况，直接返回

    # 如果 data 是 Tensor → 包两层 list
    if torch.is_tensor(data):
        data = [[data]]
    # 如果 data 是 list[tensor] → 包一层 list
    elif isinstance(data, list) and len(data) > 0 and torch.is_tensor(data[0]):
        data = [data]
    # 如果 data 已经是 list[list[tensor]] → 不动

    return (data, label)

class ComboLoader(object):
    """包装多个 DataLoader，每次返回一个合并的 batch"""
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    def combine_batch(self, batches):
        # 返回前修正结构，保证稳定
        safe_batches = [safe_structure(b) for b in batches]
        return safe_batches

def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    """
    计算采样概率:
    - instance: 按样本数量比例
    - class: 按类别均衡
    - sqrt/cbrt: 其他变种
    - prog: 渐进式 (从不平衡过渡到均衡)
    """
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5
    elif mode == 'cbrt':
        q = 0.125
    elif mode == 'prog':
        assert ep is not None and n_eps is not None, "prog 模式需要 ep/n_eps"
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else:
        raise ValueError("无效的 mode")

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)
    return sampling_probabilities

def modify_loader(loader, mode, ep=None, n_eps=None):
    """
    根据采样模式 (instance/class/prog) 返回一个新的 DataLoader
    """
    class_count = np.unique(loader.dataset.dr, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    sample_weights = sampling_probs[loader.dataset.dr]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return mod_loader

def get_combo_loader(loader, base_sampling='instance'):
    """
    构造一个组合 loader，里面有两份:
    - 一个不平衡 (instance-based)
    - 一个均衡 (class-based)
    """
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(loader, mode=base_sampling)

    balanced_loader = modify_loader(loader, mode='class')
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader
