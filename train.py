#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name:     train.py
# Author:        Run yang
# Created Time:  2024-11-21  09:16
# Last Modified: <none>-<none>

import argparse
import json
import math
import os
import time
import shutil
import traceback
import math
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from accelerate import DeepSpeedPlugin
from transformers import LlamaConfig, LlamaForCausalLM

from component.utils import setup_logger
from component.dataset.dataset import preprocess
from component.dataset.dataset import prepare_dataloader, DataCollatorForPockerDataset
from component.tokenizer.tokenizer import SP_TOKEN


def main(cfg):
    plugin = DeepSpeedPlugin(hf_ds_config=cfg.config_path)
    accelerator = Accelerator(deepspeed_plugin=plugin)

    logger = setup_logger('octopus', save_dir=cfg.workspace, distributed_rank=dist.get_rank())
    os.makedirs(cfg.workspace, exist_ok=True)
    logger.info(accelerator.state)
    
    # If passed along, set the training seed now.

    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)


    train_dataset = load_from_disk(cfg.train_dataset_path)
    num_processes = os.cpu_count()  # 使用所有可用的CPU核心数
    local_rank = int(os.environ['LOCAL_RANK'])

    # preprocess
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(preprocess,
                                        batched=True, num_proc=num_processes,
                                        remove_columns=["states", "actions", "targets", "trajectory"],
                                        # load_from_cache_file=False
                                        )
    # if local_rank > 0:
    #     dist.barrier()
    # if local_rank == 0:
    #     dist.barrier()
    
    
   
    data_collator = DataCollatorForPockerDataset(nearest_base=4)

    logger.info('start load model')

    from transformers import LlamaConfig, LlamaModel


    hidden_size = cfg.hidden_size
    # 中间层取 8/3 倍，按 128 向上取整
    intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128
    # 自定义配置
    config = LlamaConfig(
        vocab_size=128,          # 词表大小
        hidden_size=hidden_size,           # 隐藏层维度
        num_hidden_layers=8,       # Transformer 层数
        num_attention_heads=8,     # 注意力头数
        intermediate_size=intermediate_size,    # FFN 中间层维度
        max_position_embeddings=512,# 最大序列长度
        pad_token_id=SP_TOKEN["PAD"]
        
    )

    # 使用自定义配置初始化模型
    base_model = LlamaModel(config)
    base_model._attn_implementation = "flash_attention_2"
    base_model.gradient_checkpointing_enable() 
    base_model.train()
    logger.info('load model done')
    
    from component.model.model import ActorModel
    
    Qnet = ActorModel(base_model, hidden_size=hidden_size)
    Qnet.train()

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=plugin.hf_ds_config.config['train_micro_batch_size_per_gpu']
    )


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    num_update_steps_per_epoch = len(train_dataloader)

    optim_params = [p for n, p in Qnet.named_parameters() if p.requires_grad]

    optimizer_cls = DummyOptim
    optimizer = optimizer_cls(optim_params, lr=cfg.learning_rate)


    # Scheduler and math around the number of training steps.
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (plugin.hf_ds_config.config['gradient_accumulation_steps'] * dist.get_world_size()))
    max_train_steps = cfg.max_epochs * num_update_steps_per_epoch
    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    logger.info(f' warmup ratio = {math.ceil(max_train_steps * cfg.warmup_ratio)}')
    
    lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=math.ceil(max_train_steps * cfg.warmup_ratio)
        )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        Qnet, optimizer, train_dataloader, lr_scheduler)
    
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.max_epochs}")
    logger.info(f"  Instantaneous batch size per device = {plugin.hf_ds_config.config['train_micro_batch_size_per_gpu']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    completed_steps = 0
    starting_epoch = 0
    from collections import deque
    times = deque(maxlen=10)
    
    def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.AVG)
        return tensor
    # update progress bar if resumed from checkpoint
    
    for epoch in range(starting_epoch, cfg.max_epochs):
        model.train()

        active_dataloader = train_dataloader
        for _, batch in enumerate(active_dataloader):
            model.train()
            batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            start_time = time.time()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs['loss']
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                end_time = time.time()
                step_time = end_time - start_time
                times.append(step_time)
                avg_time = sum(times) / len(times)
                step_loss = all_reduce_mean(loss.detach().clone()).item()
                completed_steps += 1
                logger.info(f"""Total steps = {max_train_steps},
                                epoch: {epoch},
                                completed_steps: {completed_steps},
                                loss: {step_loss},
                                acc: {outputs['acc']},
                                """.strip())

                accelerator.log(
                    {
                        "train_loss": step_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                        "acc": outputs['acc']
                    },
                    step=completed_steps,
                )

                # if accelerator.is_main_process:
                torch.cuda.empty_cache()
    
    accelerator.wait_for_everyone()
    logger.info('Training finish')
    logger.info('Start saving model')
    # booster.save_model(model, save_path, shard=True)

    if accelerator.is_main_process:
        full_model_path = os.path.join(save_path, 'model')
        os.makedirs(full_model_path, exist_ok=True)
        
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.wait_for_everyone()
    accelerator.save({
                "model": unwrapped_model.state_dict()
            }, full_model_path)
        
    accelerator.wait_for_everyone()
    accelerator.end_training()
    



if __name__ == "__main__":
    # 
    parser = argparse.ArgumentParser(description="Qnet 4 doudizhu")
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/mnt/dhsys/doudizhu/final_dataset-reward-shape",
        help="train_data path"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='/mnt/dhsys/doudizhu/training/exp1',
        help="save_path"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default='/mnt/dhsys/doudizhu/training/',
        help="workspace"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default='/mnt/dhsys/doudizhu/training/ds_config.json',
        help="ds config path",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="max_epochs",
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="warmup_ratio",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="learning_rate",
    )
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="hidden size",
    )

    args = parser.parse_args()
    main(args)


