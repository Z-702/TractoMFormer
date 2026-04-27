#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


class FusionModel(torch.nn.Module):
    def __init__(self, model_list, cfg):
        super(FusionModel, self).__init__()
        self.model_list = torch.nn.ModuleList(model_list)
        self.cfg = cfg

    def forward(self, xs):
        pred_list = []
        for i, model in enumerate(self.model_list):
            pred_list.append(model(xs[i]))
        return torch.mean(torch.stack(pred_list, 0), 0)


def build_model(cfg, gpu_id=None):
    """
    Builds the model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in mvit/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    name = cfg.MODEL.MODEL_NAME
    model_list = [MODEL_REGISTRY.get(name)(cfg) for _ in range(cfg.DATA_NUM)]
    model = FusionModel(model_list, cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        model = model.cuda(device=cur_device)

    if cfg.NUM_GPUS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model