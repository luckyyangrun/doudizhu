#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name:     model.py
# Author:        Run yang
# Created Time:  2024-11-21  08:38
# Last Modified: <none>-<none>

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from component.tokenizer.tokenizer import SP_TOKEN, ING_TOKEN

class ActorModel(nn.Module):
    
    def __init__(self, pretrained_model, hidden_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = pretrained_model
        self.qhead = nn.Linear(hidden_size, 2)
        self.qhead.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))
    
    def forward(self,
                input_ids: torch.LongTensor,
                labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # including loss calculation
        # input_ids: [b, s]
        # labels: [b, s]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        
        q_value = self.qhead(last_hidden_states)
        
        mask = labels != ING_TOKEN
        # import pdb;pdb.set_trace()
        q_value = torch.masked_select(q_value, mask.unsqueeze(-1).repeat(1, 1, 2))
        q_value = q_value.view(-1, 2)
        labels = torch.masked_select(labels, mask)
        labels = labels.view(-1)
        labels[labels==-1] +=1
        # 
        loss = F.cross_entropy(q_value, labels, reduction="mean")
        with torch.no_grad():
            predictions = torch.argmax(q_value, dim=1)  # 在最后一维选择概率最大的类别

            # 比较预测类别和真实标签
            correct_predictions = (predictions == labels).sum().item()  # 计算预测正确的数量
            total_predictions = labels.size(0)  # 总的样本数

            # 计算准确率
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else -1
        
        return {"loss": loss, "acc": accuracy}
        
        