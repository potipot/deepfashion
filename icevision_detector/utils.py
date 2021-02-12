from collections import Counter
from typing import Optional, Union

import matplotlib.pyplot as plt

import torch
from icevision import ClassMap


def plot_counter(counter: Counter):
    fig, axs = plt.subplots(1)
    axs.bar(x=list(counter.keys()), height=counter.values())
    plt.setp(axs.xaxis.get_majorticklabels(), rotation=-90)
    plt.show()


def plot_classes_histogram(train_records, class_map: ClassMap = None, return_counter=False):
    counter = Counter()
    for record in train_records:
        counter.update(record.labels)
    if class_map: counter = {class_map.id2class[k]: v for k, v in counter.items()}
    plot_counter(counter)
    return counter if return_counter else None


def zero_infinity(t: torch.Tensor) -> torch.Tensor:
    # this preserves the dtype of the tensor
    t[t != t] = torch.tensor([0.0])
    return t


def get_pl_accelerator(gpus: Union[str, tuple, list, int]) -> Optional[str]:
    if isinstance(gpus, (tuple, list)): multi_gpu = len(gpus) > 1
    elif isinstance(gpus, int): multi_gpu = gpus > 1
    elif isinstance(gpus, str): raise NotImplementedError
    else: raise ValueError
    return 'ddp' if multi_gpu else None

