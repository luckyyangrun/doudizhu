#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name:     dataset.py
# Author:        Run yang
# Created Time:  2024-11-21  07:12
# Last Modified: <none>-<none>


import torch
import numpy as np
import random
from typing import Iterator, Optional
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed import ProcessGroup
import torch.nn.functional as F

from component.tokenizer.tokenizer import SP_TOKEN, ING_TOKEN


def preprocess(examples):
    traj = examples['trajectory']
    labels = examples['targets']
    attention_mask = [[1]* len(label) for label in labels]
    return dict(input_ids=traj, labels=labels, attention_mask=attention_mask)


class StatefulDistributedSampler(DistributedSampler):

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index:]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def prepare_dataloader(dataset,
                       batch_size,
                       shuffle=False,
                       seed=1024,
                       drop_last=False,
                       pin_memory=False,
                       num_workers=0,
                       process_group: Optional[ProcessGroup] = None,
                       **kwargs):
    r"""
    sss
    """
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()

    sampler = StatefulDistributedSampler(dataset,
                                             num_replicas=process_group.size(),
                                             rank=process_group.rank(),
                                             shuffle=shuffle)

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      worker_init_fn=seed_worker,
                      drop_last=drop_last,
                      pin_memory=pin_memory,
                      num_workers=num_workers,
                      **_kwargs)
    
def find_nearest_multiple(a, base):
    """
    找到最接近的基数倍数
    
    Args:
        a (float): 需要找到最接近的基数倍数的数字
        base (float): 基数
    
    Returns:
        float: 最接近的基数倍数
 
    """
    next_multiple = ((a + base - 1) // base) * base
    return next_multiple



def reverse_and_pad_sequence(sequences, batch_first=True, padding_value=0):
    """
    Reverse sequences, pad on the right, then reverse back to achieve left padding.

    Args:
        sequences (List[Tensor]): List of tensors to pad.
        batch_first (bool): If True, output will have batch size as the first dimension.
        padding_value (int): The value to use for padding.

    Returns:
        Tensor: Padded tensor with left-side padding.
    """
    # Reverse each sequence
    reversed_sequences = [seq.flip(0) for seq in sequences]

    # Pad the reversed sequences on the right
    padded_reversed = torch.nn.utils.rnn.pad_sequence(
        reversed_sequences, batch_first=batch_first, padding_value=padding_value
    )

    # Reverse back to achieve left padding
    return padded_reversed.flip(-1)


@dataclass
class DataCollatorForPockerDataset(object):
    """Collate examples for supervised fine-tuning."""
    nearest_base: 4


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # loneset pad
        input_ids, labels, attention_mask = tuple(
        [instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask")
        )

        # Convert to tensors
        input_ids = [torch.tensor(input_id, dtype=torch.long) for input_id in input_ids]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]

        # import pdb;pdb.set_trace()
        # Reverse and pad sequences on the left
        # input_ids = reverse_and_pad_sequence(input_ids, batch_first=True, padding_value=SP_TOKEN["PAD"])
        # labels = reverse_and_pad_sequence(labels, batch_first=True, padding_value=ING_TOKEN)
        # attention_mask = reverse_and_pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # Reverse and pad sequences on the right
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=SP_TOKEN["PAD"])
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=ING_TOKEN)

        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Add additional padding to match nearest_base
        pd_size = find_nearest_multiple(input_ids.shape[-1], self.nearest_base) - input_ids.shape[-1]
        if pd_size > 0:
            p1d = (0, pd_size)  # Padding on the right
            input_ids = F.pad(input_ids, p1d, "constant", SP_TOKEN["PAD"])
            labels = F.pad(labels, p1d, "constant", -100)
            attention_mask = F.pad(attention_mask, p1d, "constant", 0)

        return dict(input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    )
