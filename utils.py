import torch
import torch.distributed as dist

def sinkhorn(x: torch.Tensor, num_itr: int = 3, distributed: bool = True) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for optimal transport.
    
    Args:
        x: Input tensor to normalize
        num_itr: Number of normalization iterations
        distributed: Whether to sum across all devices
        
    Returns:
        Normalized tensor where each sample sums to 1
    """
    for _ in range(num_itr):
        # Sum across samples for each prototype
        weight_per_proto = torch.sum(x, dim=0, keepdim=True)
        
        if distributed and dist.is_initialized():
            # Sum across all devices
            dist.all_reduce(weight_per_proto)
            
        # Normalize by prototype sums
        x = x / (weight_per_proto + 1e-8)
        
        # Sum across prototypes for each sample
        weight_per_sample = torch.sum(x, dim=-1, keepdim=True)
        
        # Normalize by sample sums
        x = x / (weight_per_sample + 1e-8)
    
    return x


class MockWriter:
    def __init__(self):
        pass

    def add_scalar(self, name, value, step):
        pass

    def flush(self):
        pass

    def close(self):
        pass