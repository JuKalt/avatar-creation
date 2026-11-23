"""LOD-aware model; all unchanged computation is inherited from MMLPHuman.

The upstream checkout must be on sys.path, as in MMLPHuman's own entry points.
Parameter storage always remains full-sized. Only forward computations gather
rows, so the original optimizer groups and full checkpoint tensors are retained.
"""

import copy
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from scene.gaussian_model import GaussianModel as OriginalGaussianModel
from scene.gaussian_model import spherical_harmonics

from .lod import LODSelection, validate_fraction


def cached(function):
    @wraps(function)
    def getter(self):
        if function.__name__ not in self.cache_dict:
            self.cache_dict[function.__name__] = function(self)
        return self.cache_dict[function.__name__]
    return property(getter)


class LODGaussianModel(OriginalGaussianModel):
    checkpoint_type = "mmlphuman-lod"
    checkpoint_version = 1

    def __init__(self, min_fraction=0.1, max_fraction=1.0, seed=0):
        super().__init__()
        self.lods = LODSelection(min_fraction, max_fraction, seed)
        self.training = False
        self.eval_lod_fraction = 1.0
        self.lod_fraction = 1.0
        self.lod_indices = None

    def _select(self, tensor):
        return tensor if self.lod_indices is None else tensor[self.lod_indices]

    def set_lod(self, fraction):
        self.lod_fraction = validate_fraction(fraction)
        self.lod_indices = self.lods.indices(self.lod_fraction)
        self.cache_dict = {}

    def _extend_encoder(self):
        """Preserve initialization and RNG state; start the extra input at zero."""
        weight = self.encoder_feat_params["layers.0.weight"]
        if weight.shape[-1] != 63:
            raise ValueError("Expected the original 63-dimensional MMLPHuman pose encoder.")
        self.encoder_feat_params["layers.0.weight"] = nn.Parameter(
            torch.cat((weight.detach(), torch.zeros_like(weight[..., :1])), dim=-1)
        )
        self.encoder_feat_model_meta = copy.deepcopy(self.encoder_feat_model_meta)
        first = self.encoder_feat_model_meta.layers[0]
        self.encoder_feat_model_meta.layers[0] = nn.Linear(
            64, first.out_features, bias=first.bias is not None,
            device="meta", dtype=first.weight.dtype,
        )

    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self._extend_encoder()
        self.lods.initialize(len(self._xyz), self._xyz.device)
        self.set_lod(1.0)

    def training_setup(self, args, scene_scale):
        super().training_setup(args, scene_scale)
        self.training = True

    @property
    def get_joint_features(self):
        pose = super().get_joint_features
        return torch.cat((pose, pose.new_tensor([self.lod_fraction])))

    @property
    def get_weights(self):
        return self._select(super().get_weights)

    @cached
    def get_encoded_feature_gsparam_weight(self):
        features = self.get_encoded_feature[..., :self.num_basis]
        return torch.einsum(
            "nrc,nr->nc", features[self._select(self.nbr_gsft)],
            self._select(self.nbr_gsft_wght),
        )

    @cached
    def get_dxyz(self):
        weights = self._select(self.nbr_gs_invdist)
        displacements = self.get_dxyz_vt[self._select(self.nbr_gs)]
        return (weights[..., None] * displacements).sum(dim=1) / weights.sum(dim=-1)[..., None]

    @cached
    def get_cano_xyz(self):
        return self._select(self._xyz) + self.get_dxyz + torch.tanh(self._select(self.xyz_offset)) * 0.008

    @cached
    def get_cano_scaling(self):
        scaling = self._select(self._scaling)
        if self.is_gsparam_bs:
            scaling = scaling + torch.einsum(
                "nc,ncl->nl", self.get_encoded_feature_gsparam_weight, self._select(self.scaling_bs)
            )
        return self.scaling_activation(scaling)

    @property
    def get_cano_rotation(self):
        rotation = self._select(self._rotation)
        if self.is_gsparam_bs:
            rotation = rotation + torch.einsum(
                "nc,ncl->nl", self.get_encoded_feature_gsparam_weight, self._select(self.rotation_bs)
            )
        return self.rotation_activation(rotation)

    @property
    def get_opacity(self):
        opacity = self._select(self._opacity)
        if self.is_gsparam_bs:
            opacity = opacity + torch.einsum(
                "nc,nc->n", self.get_encoded_feature_gsparam_weight, self._select(self.opacity_bs)
            )
        return self.opacity_activation(opacity)

    @cached
    def get_sh(self):
        sh = self._select(self._sh0)
        if self.sh_degree > 0:
            sh = torch.cat((sh, self._select(self._shN)), dim=1)
        if self.is_gsparam_bs:
            features = self.get_encoded_feature_gsparam_weight
            delta = torch.einsum("nc,ncxy->nxy", features, self._select(self.sh0_bs))
            if self.sh_degree > 0:
                delta = torch.cat((delta, torch.einsum(
                    "nc,ncxy->nxy", features, self._select(self.shN_bs)
                )), dim=1)
            sh = sh + delta
        return sh

    def get_color(self, cam_pos):
        # Camera-dependent colors must never survive a camera change.
        if self.sh_degree > 0:
            directions = F.normalize(cam_pos - self.get_xyz, dim=-1)
            directions = torch.einsum(
                "nij,nj->ni", self.get_Gweights[:, :3, :3].transpose(-1, -2), directions
            )
        else:
            directions = torch.ones_like(self._select(self._xyz))
        return torch.clamp_min(spherical_harmonics(self.sh_degree, directions, self.get_sh) + 0.5, 0)

    def render(self, cam, override_color=None, scaling_modifier=1.0, background=None,
               *, lod_fraction=None):
        if lod_fraction is None:
            lod_fraction = (
                self.lods.sample() if self.training and torch.is_grad_enabled()
                else self.eval_lod_fraction
            )
        self.set_lod(lod_fraction)
        if override_color is not None:
            # Overrides use the active permutation order, including at 100%.
            # Callers supplying overrides should fix the LOD explicitly.
            if len(override_color) != len(self.lod_indices):
                raise ValueError("Override colors must match the active subset in permutation order.")
        return super().render(cam, override_color, scaling_modifier, background)

    def capture(self):
        data = super().capture()
        # The upstream capture reads get_weights, which is subset-aware here.
        data["_weights"] = super().get_weights
        data["lod"] = {
            "model_type": self.checkpoint_type,
            "version": self.checkpoint_version,
            "config": dict(self.lods.config),
            "permutation": self.lods.permutation,
        }
        return data

    def restore(self, data):
        lod = data.get("lod", {})
        if (lod.get("model_type") != self.checkpoint_type
                or lod.get("version") != self.checkpoint_version):
            raise ValueError("Expected an mmlphuman-lod v1 checkpoint; use restore_legacy explicitly to convert a baseline.")
        if data["encoder_feat_params"]["layers.0.weight"].shape[-1] != 64:
            raise ValueError("LOD checkpoints require a 64-dimensional pose encoder.")
        self.lod_indices = None
        super().restore(data)
        self.lods = LODSelection(**lod["config"])
        self.lods.restore(lod["permutation"], len(self._xyz), self._xyz.device)
        self.set_lod(1.0)

    def restore_legacy(self, data):
        """Explicit conversion, preserving the baseline encoder at every fraction."""
        if "lod" in data:
            raise ValueError("Use restore() for an LOD checkpoint.")
        self.lod_indices = None
        data = dict(data)
        data["encoder_feat_params"] = dict(data["encoder_feat_params"])
        super().restore(data)
        self._extend_encoder()
        self.lods.initialize(len(self._xyz), self._xyz.device)
        self.set_lod(1.0)

    def lod_metadata(self):
        return {
            "lod_percent": self.lod_fraction * 100,
            "splat_count": len(self.lod_indices),
            "total_splat_count": len(self._xyz),
        }
