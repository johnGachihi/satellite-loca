from pathlib import Path

import ml_collections
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from engine_pretrain import train_epoch
from config import get_config
from vit import create_ViTLOCAModel
import copy

def train(
        *,
        config: ml_collections.ConfigDict,
        model: nn.Module,
        dataset: Any,
        workdir: str,
        writer: Any
)  -> Tuple[nn.Module, nn.Module, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    # Initialize models
    model = model.to(device)
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[device])
    ema_model = copy.deepcopy(model)

    # Create optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.ndim == 1:  # Bias and LayerNorm parameters
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=config.learning_rate)

    # Initialize training variables
    global_step = 0

    # Restore checkpoint if exists
    if config.checkpoint:
        checkpoint_path = Path(workdir) / 'checkpoint.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            ema_model.load_state_dict(checkpoint['ema_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_step = checkpoint['global_step']

    # Training loop
    total_steps = len(dataset.train_loader) * config.num_epochs
    all_metrics = []

    for epoch in range(config.num_epochs):
        global_step, epoch_metrics = train_epoch(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            data_loader=dataset.train_loader,
            config=config,
            writer=writer,
            device=device,
            global_step=global_step,
            epoch=epoch,
            total_steps=total_steps
        )
        all_metrics.extend(epoch_metrics)

        # Checkpoint at end of epoch
        if is_main_process:
            checkpoint = {
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step
            }
            torch.save(checkpoint, Path(workdir) / 'checkpoint.pt')
            
    return model, ema_model, all_metrics


if __name__ == '__main__':
    config = get_config()

    model = create_ViTLOCAModel(config)
    dataset = LOCADataset(config)

    train(
        config=config,
        dataset=
        
