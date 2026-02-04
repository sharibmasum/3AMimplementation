import torch
from torch import nn


class MUSt3RLikeGeometryExtractor(nn.Module):
    """
    Lightweight MUSt3R-like geometric feature extractor.

    This runs in parallel with the SAM2 image encoder using RGB-only inputs.
    It outputs multi-level features:
    - early_semantic: higher-resolution, appearance-aligned features
    - later_geometric: lower-resolution, geometry-biased features
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dims=(64, 128, 256),
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev_dim = in_channels
        for dim in feature_dims:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_dim, dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.GELU(),
                )
            )
            prev_dim = dim

    def forward(self, rgb: torch.Tensor):
        features = []
        x = rgb
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class FeatureMerger(nn.Module):
    """
    Fuse SAM2 appearance features with multi-level geometry features.

    Fusion:
    - cross-attention over each geometry level
    - convolutional refinement on the fused spatial feature map
    """

    def __init__(
        self,
        appearance_dim: int,
        geometry_dims,
        num_heads: int = 8,
        refinement_depth: int = 2,
    ):
        super().__init__()
        if not isinstance(geometry_dims, (list, tuple)) or len(geometry_dims) == 0:
            raise ValueError("geometry_dims must be a non-empty list/tuple.")
        self.geom_projs = nn.ModuleList(
            [nn.Conv2d(dim, appearance_dim, kernel_size=1) for dim in geometry_dims]
        )
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(appearance_dim, num_heads) for _ in geometry_dims]
        )
        refinement_layers = []
        for _ in range(refinement_depth):
            refinement_layers.append(
                nn.Sequential(
                    nn.Conv2d(appearance_dim, appearance_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, appearance_dim),
                    nn.GELU(),
                )
            )
        self.refine = nn.Sequential(*refinement_layers)

    def forward(self, appearance_features, geometry_features):
        if geometry_features is None or len(geometry_features) == 0:
            return appearance_features
        if len(geometry_features) != len(self.geom_projs):
            raise ValueError(
                "geometry_features length must match geometry_dims used at init."
            )

        b, c, h, w = appearance_features.shape
        query = appearance_features.flatten(2).permute(2, 0, 1)
        fused = query
        for geom, proj, attn in zip(geometry_features, self.geom_projs, self.cross_attn):
            geom = proj(geom)
            key = geom.flatten(2).permute(2, 0, 1)
            fused, _ = attn(query=fused, key=key, value=key)

        fused_map = fused.permute(1, 2, 0).view(b, c, h, w)
        fused_map = self.refine(fused_map)
        return fused_map

