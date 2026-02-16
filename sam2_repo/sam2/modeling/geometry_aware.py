import inspect
import os
import torch
from torch import nn
import torch.nn.functional as F

from sam2.modeling.position_encoding import PositionEmbeddingSine
from omegaconf import ListConfig

try:
    from must3r.engine.inference import postprocess
    from must3r.tools.image import get_resize_function, unpatchify
    from dust3r.utils.image import ImgNorm
except Exception:
    postprocess = None
    get_resize_function = None
    unpatchify = None
    ImgNorm = None


class MUSt3RLikeGeometryExtractor(nn.Module):
    """
    Lightweight MUSt3R-like geometric feature extractor.

    This runs in parallel with the SAM2 image encoder using RGB-only inputs.
    It outputs multi-level features in a MUSt3R-compatible dictionary:
    - features: list of tensors, shallow -> deep
    - pos_2d: optional 2D positional encodings per level
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dims=(64, 128, 256),
        return_dict: bool = True,
        pos_enc_dim: int = 256,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.return_dict = return_dict
        self.pos_enc = (
            PositionEmbeddingSine(num_pos_feats=pos_enc_dim)
            if return_dict
            else None
        )
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
        pos_2d = []
        x = rgb
        for block in self.blocks:
            x = block(x)
            features.append(x)
            if self.pos_enc is not None:
                pos_2d.append(self.pos_enc(x))
        if not self.return_dict:
            return features
        outputs = {"features": features}
        if pos_2d:
            outputs["pos_2d"] = pos_2d
        return outputs


class MUSt3RGeometryExtractor(nn.Module):
    """
    Wrapper around an external MUSt3R model to extract geometry features.

    Expected outputs (by convention):
    - "features": list of multi-level features (encoder/decoder)
    - "pos_2d": list of 2D positional encodings for each feature
    - "point_map": per-pixel 3D point map (B, 3, H, W)
    - "ray_map": per-pixel ray map (B, 3, H, W)
    """

    def __init__(
        self,
        must3r_model: nn.Module,
        feature_indices=(0, 4, 7, 11),
        freeze: bool = True,
        sam2_img_mean=(0.485, 0.456, 0.406),
        sam2_img_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.must3r_model = must3r_model
        self.feature_indices = list(feature_indices)
        self.sam2_img_mean = torch.tensor(sam2_img_mean).view(1, 3, 1, 1)
        self.sam2_img_std = torch.tensor(sam2_img_std).view(1, 3, 1, 1)
        self._debug_logged = False
        if freeze:
            for param in self._get_modules():
                for p in param.parameters():
                    p.requires_grad = False
                param.eval()

    def _select_features(self, outputs):
        if "features" in outputs:
            features = outputs["features"]
        elif "decoder_features" in outputs:
            features = outputs["decoder_features"]
        elif "encoder_features" in outputs:
            features = outputs["encoder_features"]
        else:
            raise KeyError(
                "MUSt3R outputs must include 'features', 'decoder_features', or 'encoder_features'."
            )
        if len(features) == 0:
            return []
        max_idx = len(features) - 1
        indices = [min(idx, max_idx) for idx in self.feature_indices]
        return [features[idx] for idx in indices]

    def _get_modules(self):
        if isinstance(self.must3r_model, dict):
            return [self.must3r_model["encoder"], self.must3r_model["decoder"]]
        if isinstance(self.must3r_model, (list, tuple)):
            return list(self.must3r_model)
        return [self.must3r_model]

    def _prepare_inputs(self, rgb: torch.Tensor):
        if get_resize_function is None or ImgNorm is None:
            return rgb, torch.tensor([[rgb.shape[-2], rgb.shape[-1]]], device=rgb.device)

        device = rgb.device
        mean = self.sam2_img_mean.to(device)
        std = self.sam2_img_std.to(device)
        rgb = rgb * std + mean  # undo SAM2 normalization -> [0,1]
        rgb = rgb.clamp(0.0, 1.0)
        rgb = (rgb - 0.5) / 0.5  # MUSt3R normalization

        image_size = None
        if isinstance(self.must3r_model, dict):
            image_size = self.must3r_model.get("image_size")
            encoder = self.must3r_model["encoder"]
        elif isinstance(self.must3r_model, (list, tuple)):
            encoder = self.must3r_model[0]
            image_size = getattr(encoder, "img_size", None)
        else:
            encoder = None

        patch_size = getattr(encoder, "patch_size", 16)
        resized = []
        true_shapes = []
        for img in rgb:
            h, w = img.shape[-2:]
            resize_op, _, _ = get_resize_function(image_size or max(h, w), patch_size, h, w)
            img_cpu = img.detach().cpu()
            img_resized = resize_op(img_cpu)
            true_shapes.append(torch.tensor([img_resized.shape[-2], img_resized.shape[-1]]))
            resized.append(img_resized)
        rgb_resized = torch.stack(resized, dim=0).to(device)
        true_shape = torch.stack(true_shapes, dim=0).to(device)
        return rgb_resized, true_shape

    def _run_bundle(self, rgb: torch.Tensor):
        if postprocess is None or unpatchify is None:
            raise RuntimeError("MUSt3R helpers not available; check must3r installation.")
        bundle = self.must3r_model
        encoder = bundle["encoder"]
        decoder = bundle["decoder"]

        rgb, true_shape = self._prepare_inputs(rgb)

        x_list, pos_list = [], []
        for img, ts in zip(rgb, true_shape):
            x, pos = encoder(img.unsqueeze(0), ts.unsqueeze(0))
            x_list.append(x)   # [1, N, D]
            pos_list.append(pos)

        x = torch.stack(x_list, dim=0)    # [B, 1, N, D]
        pos = torch.stack(pos_list, dim=0)
        true_shape = true_shape.view(true_shape.shape[0], 1, 2)

        decoder_out = decoder(x, pos, true_shape, return_feats=True)
        if len(decoder_out) == 3:
            _, pointmaps, feats = decoder_out
        else:
            _, pointmaps = decoder_out
            feats = None
        pointmaps_i = pointmaps[:, 0]
        pp = postprocess(pointmaps_i)
        point_map = pp.get("pts3d")
        ray_map = pp.get("pts3d_local")
        if point_map is not None and point_map.dim() == 4 and point_map.shape[-1] == 3:
            point_map = point_map.permute(0, 3, 1, 2).contiguous()
        if ray_map is not None and ray_map.dim() == 4 and ray_map.shape[-1] == 3:
            ray_map = ray_map.permute(0, 3, 1, 2).contiguous()

        feats_i = feats if feats is not None else []
        feat_maps = []
        for feat in feats_i:
            feat_tokens = feat[0, 0].unsqueeze(0)  # (1, N, D) for first item in batch
            feat_map = unpatchify(
                feat_tokens,
                encoder.patch_size,
                true_shape[0, 0].tolist(),
            )
            if (
                feat_map.dim() == 4
                and feat_map.shape[-1] > feat_map.shape[1]
                and feat_map.shape[-1] > feat_map.shape[-2]
            ):
                # MUSt3R can return channel-last features; convert to channel-first.
                feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
            feat_maps.append(feat_map)
        return {
            "features": feat_maps,
            "pos_2d": None,
            "point_map": point_map,
            "ray_map": ray_map,
        }

    def _run_model(self, rgb: torch.Tensor):
        if isinstance(self.must3r_model, dict):
            return self._run_bundle(rgb)
        # Prefer any dedicated helpers if the MUSt3R API provides them.
        for method_name in ("forward_features", "inference", "infer", "encode"):
            method = getattr(self.must3r_model, method_name, None)
            if callable(method):
                return method(rgb)

        # Fallback: call forward with supported args.
        forward_fn = self.must3r_model.forward
        sig = inspect.signature(forward_fn)
        kwargs = {}
        if "true_shape" in sig.parameters:
            kwargs["true_shape"] = rgb.shape[-2:]
        if "pos" in sig.parameters:
            get_pos = getattr(self.must3r_model, "get_pos", None)
            kwargs["pos"] = get_pos(rgb) if callable(get_pos) else None
        try:
            return forward_fn(rgb, **kwargs)
        except TypeError:
            return forward_fn(rgb)

    def forward(self, rgb: torch.Tensor):
        with torch.no_grad():
            outputs = self._run_model(rgb)
        if not self._debug_logged:
            self._debug_logged = True
            print(
                "MUSt3RGeometryExtractor forward:",
                f"input={tuple(rgb.shape)}",
                f"device={rgb.device}",
                f"features={len(outputs.get('features', []))}",
            )
        features = self._select_features(outputs)
        if os.getenv("MUST3R_DEBUG", "0") == "1":
            shapes = [tuple(f.shape) for f in features]
            channels = [f.shape[1] if f.dim() == 4 else None for f in features]
            print("MUSt3R selected feature shapes:", shapes)
            print("MUSt3R selected feature channels:", channels)
        if os.getenv("REQUIRE_MUST3R", "0") == "1" and len(features) == 0:
            raise RuntimeError("MUSt3R produced no features; geometry fusion is inactive.")
        return {
            "features": features,
            "pos_2d": outputs.get("pos_2d"),
            "point_map": outputs.get("point_map"),
            "ray_map": outputs.get("ray_map"),
        }


class _AttnFFNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key, q_pos=None, k_pos=None):
        q = query if q_pos is None else query + q_pos
        k = key if k_pos is None else key + k_pos
        attn_out, _ = self.attn(q, k, k)
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class FeatureMerger(nn.Module):
    """
    Fuse SAM2 appearance features with multi-level geometry features.

    Mechanics:
    - self-attention on shallow MUSt3R feature
    - sequential cross-attention with deeper MUSt3R layers
    - FFN after each attention block
    - convolutional refinement
    - concatenate with SAM2 FPN output, then final conv -> Fmerged
    """

    def __init__(
        self,
        appearance_dim: int,
        geometry_dims,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        refinement_depth: int = 2,
    ):
        super().__init__()
        if isinstance(geometry_dims, ListConfig):
            geometry_dims = list(geometry_dims)
        if geometry_dims is None or len(geometry_dims) == 0:
            geometry_dims = [64, 128, 256]
        if not isinstance(geometry_dims, (list, tuple)) or len(geometry_dims) == 0:
            raise ValueError("geometry_dims must be a non-empty list/tuple.")
        self.appearance_dim = appearance_dim
        self.geom_projs = nn.ModuleList(
            [nn.Conv2d(dim, appearance_dim, kernel_size=1) for dim in geometry_dims]
        )
        self.self_attn = _AttnFFNBlock(appearance_dim, num_heads, ffn_dim)
        self.cross_attn = nn.ModuleList(
            [
                _AttnFFNBlock(appearance_dim, num_heads, ffn_dim)
                for _ in range(len(geometry_dims) - 1)
            ]
        )
        self.pe2d_fallback = PositionEmbeddingSine(num_pos_feats=appearance_dim)
        self.pe3d_proj = nn.Conv2d(6, appearance_dim, kernel_size=1)
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
        self.out_conv = nn.Conv2d(appearance_dim * 2, appearance_dim, kernel_size=1)

    def _flatten_tokens(self, x):
        return x.flatten(2).permute(2, 0, 1)

    def _resize_to(self, x, size):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _get_pos_2d(self, pos_2d, level, feature):
        if isinstance(pos_2d, (list, tuple)):
            if level < len(pos_2d) and pos_2d[level] is not None:
                return pos_2d[level]
        if torch.is_tensor(pos_2d):
            return pos_2d
        return self.pe2d_fallback(feature)

    def _build_pe3d(self, point_map, ray_map, size):
        if point_map is None or ray_map is None:
            return None
        if point_map.dim() == 4 and point_map.shape[-1] == 3:
            point_map = point_map.permute(0, 3, 1, 2).contiguous()
        if ray_map.dim() == 4 and ray_map.shape[-1] == 3:
            ray_map = ray_map.permute(0, 3, 1, 2).contiguous()
        if point_map.shape[-2:] != size:
            point_map = self._resize_to(point_map, size)
        if ray_map.shape[-2:] != size:
            ray_map = self._resize_to(ray_map, size)
        pe3d = torch.cat([point_map, ray_map], dim=1)
        return self.pe3d_proj(pe3d)

    def forward(self, appearance_features, geometry_features):
        if geometry_features is None:
            return appearance_features
        if isinstance(geometry_features, dict):
            geom_feats = geometry_features.get("features")
            pos_2d = geometry_features.get("pos_2d")
            point_map = geometry_features.get("point_map")
            ray_map = geometry_features.get("ray_map")
        else:
            geom_feats = geometry_features
            pos_2d = None
            point_map = None
            ray_map = None

        if geom_feats is None or len(geom_feats) == 0:
            return appearance_features
        if len(geom_feats) != len(self.geom_projs):
            raise ValueError(
                "geometry_features length must match geometry_dims used at init."
            )
        if os.getenv("REQUIRE_MUST3R", "0") == "1":
            feat_shapes = [tuple(f.shape) for f in geom_feats]
            proj_shapes = [tuple(p.weight.shape) for p in self.geom_projs]
            print(
                "FeatureMerger geom feature shapes:",
                feat_shapes,
                "proj weights:",
                proj_shapes,
            )
            for feat, proj in zip(geom_feats, self.geom_projs):
                if feat.shape[1] != proj.weight.shape[1]:
                    raise RuntimeError(
                        "Geometry feature channels do not match feature_merger "
                        "geometry_dims; retraining is required."
                    )

        b, c, h, w = appearance_features.shape
        pe3d = self._build_pe3d(point_map, ray_map, (h, w))

        proj_geom = []
        for geom, proj in zip(geom_feats, self.geom_projs):
            geom = self._resize_to(proj(geom), (h, w))
            proj_geom.append(geom)

        shallow = proj_geom[0]
        pos2d = self._resize_to(self._get_pos_2d(pos_2d, 0, shallow), (h, w))
        pos = pos2d if pe3d is None else pos2d + pe3d
        tokens = self._flatten_tokens(shallow)
        pos_tokens = self._flatten_tokens(pos) if pos is not None else None
        tokens = self.self_attn(tokens, tokens, q_pos=pos_tokens, k_pos=pos_tokens)

        for level, geom in enumerate(proj_geom[1:], start=1):
            pos2d = self._resize_to(self._get_pos_2d(pos_2d, level, geom), (h, w))
            pos = pos2d if pe3d is None else pos2d + pe3d
            key_tokens = self._flatten_tokens(geom)
            k_pos = self._flatten_tokens(pos) if pos is not None else None
            tokens = self.cross_attn[level - 1](tokens, key_tokens, k_pos=k_pos)

        fused_map = tokens.permute(1, 2, 0).view(b, c, h, w)
        fused_map = self.refine(fused_map)
        merged = self.out_conv(torch.cat([appearance_features, fused_map], dim=1))
        return merged

