import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
from typing import Any
import torch.distributed as dist

import utils


def train_epoch(
        model: nn.Module,
        ema_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        config: ml_collections.ConfigDict,
        writer: Any,
        device: torch.device,
        global_step: int,
        epoch: int,
        total_steps: int
):
    """Runs one epoch of training.
    
    Args:
        model: Main model
        ema_model: EMA model (teacher)
        optimizer: Optimizer
        data_loader: Training data loader
        config: Training configuration
        writer: Metrics writer
        device: Device to train on
        global_step: Current training step
        epoch: Current epoch number
        total_steps: Total training steps
    
    Returns:
        Updated global step and metrics for the epoch
    """
    train_metrics = []
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    for batch in data_loader:
        global_step += 1

        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}

        # Forward pass on reference view
        use_ema = config.apply_cluster_loss
        drop_moment = 'late' if config.apply_cluster_loss else 'early'  # Interesting!
        model_for_ref = ema_model if use_ema else model

        with torch.set_grad_enabled(not use_ema):
            _, r_feat_targets, r_patch_features, _ = model_for_ref(
                batch['reference'],
                seqlen=config.reference_seqlen,
                seqlen_selection=config.reference_seqlen_selection,
                drop_moment=drop_moment,
                train=True
            )

        # Forward pass on query views
        # Random-style query
        use_pe = config.apply_cluster_loss
        q_rand_loc_pred, q_rand_feat_pred, _, q_rand_idx_kept = model(
            batch['query0'],
            inputs_kv=r_patch_features,
            seqlen=config.query_max_seqlen,
            use_pe=use_pe,
            train=True
        )

        # Focal-style queries
        n_q_foc = config.dataset_configs.number_of_focal_queries
        q_foc_loc_pred, q_foc_feat_pred, _, _ = model(
            batch['queries'],
            inputs_kv=r_patch_features.repeat(n_q_foc, 1, 1),
            use_pe=use_pe,
            train=True
        )

        # Combine predictions and prepare targets
        n_pos = config.n_ref_positions
        bs = batch['reference'].shape[0]
        
        q_loc_pred = torch.cat([
            q_rand_loc_pred.reshape(-1, n_pos),
            q_foc_loc_pred.reshape(-1, n_pos)
        ], dim=0)

        q_rand_loc_targets = batch['query0_target_position'].reshape(bs, -1)
        if len(q_rand_idx_kept) < q_rand_loc_targets.shape[1]:
            q_rand_loc_targets = torch.index_select(
                q_rand_loc_targets, 1, q_rand_idx_kept)

        q_foc_loc_targets = batch['target_positions']
        q_loc_targets = torch.cat([
            q_rand_loc_targets.reshape(-1),
            q_foc_loc_targets.reshape(-1)
        ], dim=0)

        # Calculate losses
        q_r_intersect = q_loc_targets != -1
        q_loc_targets = F.one_hot(q_loc_targets.long(), n_pos)
        
        localization_loss = F.cross_entropy(
            q_loc_pred[q_r_intersect],
            q_loc_targets[q_r_intersect],
            reduction='mean'
        )

        feature_loss = torch.tensor(0.0, device=device)
        if config.apply_cluster_loss:
            # Feature prediction loss
            k = r_feat_targets.shape[-1]
            q_feat_pred = torch.cat([
                q_rand_feat_pred, q_foc_feat_pred
            ], dim=0) / config.model.temperature

            # Process feature targets
            r_feat_targets = F.softmax(r_feat_targets / config.sharpening, dim=-1)
            r_feat_targets = utils.sinkhorn(r_feat_targets)
            r_feat_targets = r_feat_targets.reshape(bs, -1, k)

            # Get targets for queries
            q_rand_feat_targets = torch.gather(
                r_feat_targets,
                1,
                q_rand_loc_targets.unsqueeze(-1).expand(-1, -1, k)
            ).reshape(-1, k)

            r_feat_targets = r_feat_targets.repeat(n_q_foc, 1, 1)
            q_foc_feat_targets = torch.gather(
                r_feat_targets,
                1,
                q_foc_loc_targets.unsqueeze(-1).expand(-1, -1, k)
            ).reshape(-1, k)

            q_feat_targets = torch.cat([
                q_rand_feat_targets, q_foc_feat_targets
            ], dim=0)

            feature_loss = F.cross_entropy(
                q_feat_pred[q_r_intersect],
                q_feat_targets[q_r_intersect],
                reduction='mean'
            )

            # me-max regularization
            avg_prediction = torch.mean(F.softmax(q_feat_pred, dim=-1), dim=0)
            if dist.is_initialized():
                dist.all_reduce(avg_prediction)
                avg_prediction /= dist.get_world_size()
            feature_loss += torch.sum(avg_prediction * torch.log(avg_prediction + 1e-10))

        # Total loss and optimization
        total_loss = localization_loss + feature_loss
        
        optimizer.zero_grad()
        total_loss.backward()

        if config.get('max_grad_norm') is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.max_grad_norm
            )

        # Update EMA model
        momentum_param = config.momentum_rate
        with torch.no_grad():
            for param, ema_param in zip(
                model.parameters(),
                ema_model.parameters()
            ):
                ema_param.data.mul_(momentum_param).add_(
                    param.data, alpha=(1 - momentum_param)
                )

        # Log metrics
        metrics = {
            'total_loss': total_loss.item(),
            'localization_loss': localization_loss.item(),
            'feature_loss': feature_loss.item()
        }
        train_metrics.append(metrics)

        # Log progress
        if global_step % config.log_every == 0 and is_main_process:
            writer.add_scalar('train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('train/localization_loss', 
                            localization_loss.item(), global_step)
            writer.add_scalar('train/feature_loss', feature_loss.item(), global_step)
            
    return global_step, train_metrics
