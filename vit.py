import math
import string
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
from timm.models.vision_transformer import Block


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
    device:
  Returns:
    Matrix of position embeddings, has shape [1, ...], where ... = x_shape[1:].
  """

    assert len(x_shape) in (4, 5), f'Unsupported input shape: {x_shape}'
    num_parts = 4 if (len(x_shape)) == 4 else 6
    channels = x_shape[-1]
    assert channels % num_parts == 0, f'channels must be multiple of num_parts'  # I don't get this

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


def token_indexes_not_to_drop(seqlen: int, n_tokens: int, seqlen_selection: str, device: torch.device = None):
    idx_kept_tokens = torch.arange(n_tokens, device=device)

    if 0 < seqlen <= n_tokens:
        if seqlen_selection in ["first", "consequtive"]:
            if seqlen_selection == "first":
                offset = 0
            else:
                offset = torch.randint(0, n_tokens - seqlen + 1, (1,), device=device).item()
            idx_kept_tokens = torch.ones(seqlen) * offset + torch.arange(seqlen)
        elif seqlen_selection == "unstructured":
            idx_kept_tokens = torch.randperm(n_tokens, device=device)[:seqlen]

    return idx_kept_tokens.long()


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
        super().__init__()
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

    def __init__(
            self,
            patches: ml_collections.ConfigDict,
            hidden_size: int,
            posembs: Tuple[int, int] = (14, 14),
            positional_embedding: str = "learned"
    ):
        super().__init__()

        self.patches = patches
        self.hidden_size = hidden_size
        self.posembs = posembs
        self.positional_embedding = positional_embedding

        self.embedding = nn.Conv2d(
            in_channels=3,  # TODO: Revisit
            out_channels=self.hidden_size,
            kernel_size=patches.size,  # Using size from ConfigDict
            stride=patches.size,  # Same stride as kernel for non-overlapping patches
            padding=0
        )

        if positional_embedding == "learned":
            std = 1.0 / math.sqrt(hidden_size)
            self.pos_embed = nn.Parameter(
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

    def forward(
            self, x: torch.Tensor, positional_embedding: str = '',
            seqlen: int = -1, seqlen_selection: str = "unstructured"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        # Patch embed then convert from NCHW to NHWC
        x = self.embedding(x).permute(0, 2, 3, 1)

        # Add positional encodings
        x = self.add_positional_embedding(x, positional_embedding)

        # Possibly drop some tokens
        idx_kept_tokens = None
        n_tokens = self.posembs[0] * self.posembs[1]
        if seqlen > 0:
            idx_kept_tokens = token_indexes_not_to_drop(
                seqlen, n_tokens, seqlen_selection, device=self.embedding.weight.device)
            if len(idx_kept_tokens) < n_tokens:
                x = torch.index_select(x, 1, idx_kept_tokens)

        return x, idx_kept_tokens


class ViT4LOCA(nn.Module):
    """Vision Transformer model for LOCA training.

    Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    n_ref_positions: Number of position in the reference view.
    apply_cluster_loss: Whether to apply the clustering loss.
    head_hidden_dim: Dimension of the hidden layer in the projection mlp.
    head_bottleneck_dim: Dimension of the bottleneck.
    head_output_dim: Dimension of the output ("number of prototypes").
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: Stochastic depth.
    posembs: Positional embedding size.
    dtype: JAX data type for activations.
    """

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            patches: ml_collections.ConfigDict,
            hidden_size: int,
            n_ref_positions: int,
            apply_cluster_loss: bool,
            head_hidden_dim: int,
            head_bottleneck_dim: int,
            head_output_dim: int,
            mlp_ratio: int = 4,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            stochastic_depth: float = 0.1,
            posembs: Tuple[int, int] = (14, 14)
    ):
        super().__init__()
        # Store configuration
        self.mlp_ratio = mlp_ratio
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patches = patches
        self.hidden_size = hidden_size
        self.n_ref_positions = n_ref_positions
        self.apply_cluster_loss = apply_cluster_loss
        self.posembs = posembs
        self.head_output_dim = head_output_dim

        # Patchifier and patch tokenizer
        self.to_token = ToTokenSequence(
            patches=patches,
            hidden_size=hidden_size,
            posembs=posembs
        )

        # ViT Encoder
        self.encoder_blocks = nn.ModuleList([
            Block(
                dim=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop_path=stochastic_depth * (i / max(num_layers - 1, 1))
            )
            for i in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size)

        # Optional cluster prediction head
        if apply_cluster_loss:
            self.projection_head = ProjectionHead(
                in_dim=hidden_size,
                hidden_dim=head_hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
                output_dim=head_output_dim
            )

        # Cross-attention component
        self.cross_attention = CrossAttentionEncoderBlock(
            dim=hidden_size,
            mlp_dim=hidden_size * mlp_ratio,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            stochastic_depth=stochastic_depth
        )

        # Final layers
        self.final_norm = nn.LayerNorm(hidden_size)
        self.position_predictor = nn.Linear(hidden_size, n_ref_positions)

    def forward(
            self,
            x: torch.Tensor,
            inputs_kv: Optional[torch.Tensor] = None,
            train: bool = True,
            seqlen: int = -1,
            use_pe: bool = True,
            drop_moment: str = 'early',
            seqlen_selection: str = 'unstructured',
            debug: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Process input images through the LOCA transformer architecture.
        
        The forward pass consists of several stages:
        1. Convert input images to patch tokens
        2. Process through transformer encoder
        3. Optionally generate clustering predictions
        4. Apply cross-attention between query and reference patches
        5. Predict final positions
        """
        # Convert input images to sequence of patch tokens
        x, idx_kept_tokens = self.to_token(
            x,
            positional_embedding=None if use_pe else 'pe_not_in_use',
            seqlen=seqlen if drop_moment == 'early' else -1,
            seqlen_selection=seqlen_selection
        )

        # Process through transformer encoder blocks
        for encoder in self.encoder_blocks:
            x = encoder(x)
        x = self.encoder_norm(x)

        # Generate clustering predictions if requested
        cluster_pred_outputs = None
        if self.apply_cluster_loss:  # TODO. Interesting! What's happening here?
            cluster_pred_outputs = self.projection_head(x, train)
            cluster_pred_outputs = cluster_pred_outputs.reshape(
                -1, self.head_output_dim)

        # Store patch representations before potential token dropping
        patches_repr = x

        # Handle late token dropping if requested
        if drop_moment == 'late':
            idx_kept_tokens = token_indexes_not_to_drop(
                seqlen, self.n_ref_positions, seqlen_selection,
                device=x.device)
            if len(idx_kept_tokens) < self.n_ref_positions:
                patches_repr = torch.index_select(patches_repr, 1, idx_kept_tokens)

        # Apply cross-attention between query and reference patches
        if inputs_kv is None:
            inputs_kv = patches_repr.clone()

        x = self.cross_attention(x, inputs_kv=inputs_kv, train=train)
        x = self.final_norm(x)
        x = self.position_predictor(x)

        return x, cluster_pred_outputs, patches_repr, idx_kept_tokens


class ProjectionHead(nn.Module):
    """Projection head.

    Attributes:
    hidden_dim: Dimension of the hidden layer in the projection mlp.
    bottleneck_dim: Dimension of the bottleneck.
    output_dim: Dimension of the output ("number of prototypes").
    normalize_last_layer: Normalize the last layer of prototypes.
    use_bn: Use batch normalizations.
    n_layers: Depth of the projection head.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 2048,
            bottleneck_dim: int = 256,
            output_dim: int = 4096,
            n_layers: int = 2
    ):
        super().__init__()

        # Create MLP layers
        self.first_layer = nn.Linear(in_dim, hidden_dim)
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ) for _ in range(1, n_layers)
        ])

        # Bottleneck layer
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)

        # Output layer with weight normalization
        self.prototypes = WeightNormLinear(bottleneck_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        """Forward pass through projection head.
        
        Args:
            x: Input tensor
            train: Whether in training mode (not used in this implementation)
            
        Returns:
            Output tensor after projection
        """
        # Apply MLP layers with residual connections
        x = self.first_layer(x)
        for layer in self.mlp_layers:
            x = layer(x)  # Residual connection around GELU

        # Bottleneck
        x = self.bottleneck(x)

        # TODO: Interesting! Why normalize?
        # L2 normalize features
        x = F.normalize(x, p=2, dim=-1)

        # Project to prototypes with weight normalization
        x = self.prototypes(x)

        return x


class WeightNormLinear(nn.Linear):
    """Linear layer with weight normalized kernel."""

    def reset_parameters(self) -> None:
        """Initialize parameters with weight normalization."""
        # First do standard initialization
        super().reset_parameters()
        # Then normalize the kernel
        with torch.no_grad():
            self.weight.div_(torch.norm(self.weight, dim=0, keepdim=True) + 1e-10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalized weights."""
        # Normalize the weight matrix
        weight = self.weight
        weight = weight / (torch.norm(weight, dim=0, keepdim=True) + 1e-10)
        return F.linear(x, weight, self.bias)


class CrossAttentionEncoderBlock(nn.Module):
    """Transformer layer with cross-attention."""

    def __init__(
            self,
            dim: int,
            mlp_dim: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            stochastic_depth: float = 0.0
    ):
        super().__init__()
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Multi-head attention (using PyTorch's implementation)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True
        )

        # MLP block
        self.mlp = MlpBlock(
            dim=dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate
        )

        # Dropout and StochasticDepth
        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path = StochasticDepth(stochastic_depth)

    def forward(
            self,
            x: torch.Tensor,
            inputs_kv: torch.Tensor,
            train: bool = True
    ) -> torch.Tensor:
        # Attention block
        assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"

        # Normalize inputs
        q = self.norm1(x)
        kv = self.norm1(inputs_kv)

        # Apply attention
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False
        )

        # Apply dropout and stochastic depth
        attn_out = self.dropout(attn_out) if train else attn_out
        x = x + self.drop_path(attn_out, train)

        # MLP block
        y = self.mlp(self.norm2(x), train)
        y = self.drop_path(y, train)
        x = x + y

        return x


class MlpBlock(nn.Module):
    """Transformer MLP block."""

    def __init__(
            self,
            dim: int,
            mlp_dim: int,
            dropout_rate: float = 0.1,
            use_bias: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim, bias=use_bias)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if use_bias:
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x) if train else x
        x = self.fc2(x)
        x = self.dropout(x) if train else x
        return x


class StochasticDepth(nn.Module):
    """Implements stochastic depth for regularization."""

    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        if not train or self.drop_rate == 0.0:
            return x

        keep_rate = 1.0 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_rate
        return x.div(keep_rate) * random_tensor


def create_ViTLOCAModel(config: ml_collections.ConfigDict):
    return ViT4LOCA(
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        patches=config.model.patches,
        hidden_size=config.model.hidden_size,
        n_ref_positions=config.n_ref_positions,
        apply_cluster_loss=config.apply_cluster_loss,
        head_hidden_dim=config.model.get('head_hidden_dim', 2048),
        head_bottleneck_dim=config.model.get('head_bottleneck_dim', 256),
        head_output_dim=config.model.get('head_output_dim', 1024),
        mlp_ratio=config.model.mlp_ratio,
        dropout_rate=config.model.dropout_rate,
        attention_dropout_rate=config.model.attention_dropout_rate,
        stochastic_depth=config.model.stochastic_depth,
        posembs=config.model.get('posembs', (14, 14))
    )