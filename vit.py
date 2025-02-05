import torch.nn as nn
import ml_collections

def get_fixed_sincos_position_embedding(x_shape: Tuple[int, ...],
                                        temperature: float = 10_000,
                                        dtype: torch.device = torch.float32,
                                        device: torch.device = None):
    """Provides a fixed position encoding for 2D and 3D coordinates.

  The embedding follows the initialisation method used in multiple papers such
  as "Attention is All You Need", https://arxiv.org/abs/1706.03762 and
  "Better plain ViT baselines for ImageNet-1k", https://arxiv.org/abs/2205.01580

  Arguments:
    x_shape: the shape of the input for which a position embedding is needed.
    temperature: Temperature parameter.
    dtype: dtype of the position encoding.
  Returns:
    Matrix of position embeddings, has shape [1, ...], where ... = x_shape[1:].
  """

    assert len(x_shape) in (4, 5), f'Unsupported input shape: {x_shape}'
    num_parts = 4 if(len(x_shape)) == 4 else 6
    channels = x_shape[-1]
    assert channels % num_parts == 0, f'channels must be multiple of num_parts' # I don't get this

    dim_t = channels // num_parts
    omega = torch.arange(dim_t, dtype=torch.float32, device=device)
    omega = omega / dim_t
    omega = 1.0 / (temperature ** omega)

    if len(x_shape) == 4:
        _, h, w, _ = x_shape

        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )

        y_enc = torch.einsum('m,d->md', y.flatten(), omega)
        x_enc = torch.einsum('m,d->md', x.flatten(), omega)

        pos_emb = [
            torch.sin(x_enc), torch.cos(x_enc),
            torch.sin(y_enc), torch.cos(y_enc)
        ]

        shape = (1, h, w, channels)
    else:
        """Not checked since won't be used"""
        
        _, t, h, w, _ = x_shape
        
        # Create 3D coordinate grids
        z, y, x = torch.meshgrid(
            torch.arange(t, device=device),
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        
        # Calculate position encodings for each dimension
        z_enc = torch.einsum('m,d->md', z.flatten(), omega)
        y_enc = torch.einsum('m,d->md', y.flatten(), omega)
        x_enc = torch.einsum('m,d->md', x.flatten(), omega)
        
        # Generate sin/cos patterns for all dimensions
        pos_emb = [
            torch.sin(z_enc), torch.cos(z_enc),
            torch.sin(x_enc), torch.cos(x_enc),
            torch.sin(y_enc), torch.cos(y_enc)
        ]
        shape = (1, t, h, w, channels)

    pe = torch.cat(pos_emb, dim=1)
    pe = pe.reshape(*shape)
    
    return pe.to(dtype)


class AddFixedSinCosPositionEmbedding(nn.Module):
    """Adds fixed sinusoidal position encodings to input features.
    
    This module implements position-dependent frequency patterns as described in
    "Attention is All You Need" and "Better plain ViT baselines for ImageNet-1k".
    It supports both 2D (images) and 3D (video) inputs.
    
    Args:
        temperature: Controls the frequency bands of position encodings
        dtype: Desired dtype for the position encodings
    """
    def __init__(self, temperature: float = 10_000, dtype: torch.dtype = torch.float32):
        super().init()
        self.temperature = temperature
        self.dtype = dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Adds position encodings to the input tensor.
        
        Args:
            inputs: Input tensor with shape [N, H, W, C] or [N, T, H, W, C]
            
        Returns:
            Input tensor with position encodings added
        """
        pe = get_fixed_sincos_position_embedding(
            x_shape=inputs.shape,
            temperature=self.temperature,
            dtype=self.dtype,
            device=inputs.device
        )
        return inputs + pe


class ToTokenSequence(nn.Module):
    """Transform a batch of views into a sequence of tokens.
    
    This module processes images into sequences of patch tokens, adding positional
    information through either learned or fixed sinusoidal embeddings.

    It includes token dropping.
    
    Args:
        patches: Configuration for patch extraction (size and other properties)
        hidden_size: Dimensionality of the token embeddings
        posembs: Size of the positional embedding grid (height, width)
        positional_embedding: Type of positional embedding ('learned' or 'sinusoidal_2d')
    """
    def __init__(self,
                 patches: ml_collections.ConfigDict,
                 hidden_size: int,
                 posembs: Tuple[int, int] = (14, 14)
                 positional_embedding: str = "learned"):
        super().init()
        self.patches = patches
        self.hidden_size = hidden_size
        self.posembs = posembs
        self.positional_embeddings = positional_embeddings

        self.embedding = nn.Conv2D(
            in_channels = 3  # TODO: Revisit
            out_channels=hidden_size,
            kernel_size=patches.size,  # Using size from ConfigDict
            stride=patches.size,  # Same stride as kernel for non-overlapping patches
            padding=0 
        )

        if positional_embedding == "learned":
            std = 1.0 / math.sqrt(hidden_size)
            self.pos_embed = nn.Parameters(
                torch.randn(1, posembs[0], posembs[1], hidden_size) * std
            )
        elif positional_embedding == "sinusoidal_2d":
            self.fixed_pos_embed = AddFixedSinCosPositionEmbedding()

    def add_positional_embedding(self,
                                 x: torch.Tensor,
                                 positional_embedding: string = ''):
        """Add positional encodings to the input patch sequence."""
        _, h, w, c = x.shape
        positional_embedding = positional_embedding or self.positional_embedding

        if positional_embedding == 'learned':
            posemb = self.pos_embed
            if (h, w) != self.posembs:
                # Handle resolution mismatch with bilinear interpolation
                posemb = F.interpolate(
                    posemb.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            x = x + posemb
        elif positional_embedding == 'sinusoidal_2d':
            x = self.fixed_pos_embed(x)

        x = x.reshape(-1, h * w, c)
        return x

    def token_indexes_not_to_drop(self, seqlen: int, n_tokens: int, seqlen_selection: str):
        device = self.embedding.weight.device
        idx_kept_tokens = torch.arange(n_tokens, device=device)

        if seqlen > 0 and seqlen <= n_tokens:
            if seqlen_selection in ["first", "consequtive"]:
                if seqlen_selection == "first":
                    offset = 0
                else:
                    offset = torch.randint(0, n_tokens - seqlen + 1, (1,), device=device).item()
                    idx_kept_tokens = torch.ones(seqlen) * offset + torch.arange(seqlen)
            elif seqlen_selection == "unstructured":
                idx_kept_tokens = torch.randperm(n_tokens, device=device)[:seqlen]
                
        return idx_kept_tokens.long()

    def forward(
            self, x: torch.Tensor, positiona_embedding: str = '',
            seqlen: int = -1, seqlen_selection: str = "unstructured"
    ) -> Tuple[torch.Tensor], Optional[torch.Tensor]:
        """Process input images into sequences of patch tokens.
        
        This method handles the complete pipeline of:
        1. Patch extraction via convolution
        2. Position encoding addition
        3. Optional token selection
        
        Args:
            x: Input tensor in NCHW format
            positional_embedding: Override default positional embedding type
            seqlen: Number of tokens to keep (-1 for all)
            seqlen_selection: Strategy for selecting tokens
            
        Returns:
            Tuple of (processed tokens, indices of kept tokens)
        """
        
