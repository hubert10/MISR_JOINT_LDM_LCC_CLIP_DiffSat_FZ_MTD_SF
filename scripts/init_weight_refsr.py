import sys

sys.path.append(".")

import torch
import torch.nn.functional as F
from typing import Dict

from utils.hparams import hparams, set_hparams
from utils.utils_dataset import read_config

from models.denoiser.unet import Unet
from models.diffusion.latent_diffusion import LatentDiffusion
from models.encoders.t_convformer import TConvFormer


# =========================================================
# Load checkpoint
# =========================================================
def load_weight(weight_path: str) -> Dict[str, torch.Tensor]:
    weight = torch.load(
        weight_path, map_location=torch.device("cpu"), weights_only=False
    )
    if "state_dict" in weight:
        weight = weight["state_dict"]

    pure_weight = {}
    for key, val in weight.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        pure_weight[key] = val

    return pure_weight


# =========================================================
# Mapping function (VERY IMPORTANT)
# =========================================================
def sd_mapping(weight_name: str) -> str:
    if weight_name.startswith("denoise_net."):
        return weight_name.replace("denoise_net.", "model.diffusion_model.")
    elif weight_name.startswith("first_stage_model."):
        return weight_name
    elif weight_name.startswith("cond_net."):
        return weight_name.replace("cond_net.", "cond_stage_model.")
    else:
        return weight_name


# =========================================================
# Analysis function
# =========================================================
def analyze_weight_mapping(scratch_weights, sd_weights):
    stats = {
        "exact_match": [],
        "same_shape_diff": [],
        "shape_mismatch": [],
        "missing": [],
        "similarity": {},
    }

    for name, model_w in scratch_weights.items():
        target_name = sd_mapping(name)

        if target_name not in sd_weights:
            stats["missing"].append(name)
            continue

        target_w = sd_weights[target_name]

        if model_w.shape != target_w.shape:
            stats["shape_mismatch"].append((name, model_w.shape, target_w.shape))
            continue

        if torch.equal(model_w, target_w):
            stats["exact_match"].append(name)
        else:
            stats["same_shape_diff"].append(name)

            try:
                sim = F.cosine_similarity(
                    model_w.flatten(), target_w.flatten(), dim=0
                ).item()
                stats["similarity"][name] = sim
            except:
                stats["similarity"][name] = None

    return stats


# =========================================================
# Weight initialization (your logic, cleaned)
# =========================================================
def initialize_weights(model, sd_weights):
    scratch_weights = model.state_dict()
    init_weights = {}

    for weight_name in scratch_weights.keys():
        target_name = sd_mapping(weight_name)

        if target_name in sd_weights:
            target_weight = sd_weights[target_name]
            model_weight = scratch_weights[weight_name]

            if model_weight.shape == target_weight.shape:
                init_weights[weight_name] = target_weight.clone()

            else:
                model_shape = model_weight.shape
                target_shape = target_weight.shape

                print("\n[Shape mismatch]")
                print(weight_name)
                print("model:", model_shape)
                print("target:", target_shape)

                # -------- Conv weights --------
                if len(model_shape) == 4:

                    # expand input channels
                    if model_shape[1] > target_shape[1]:
                        diff = model_shape[1] - target_shape[1]
                        oc, ic, h, w = target_shape

                        zero = torch.zeros(
                            (oc, diff, h, w),
                            dtype=target_weight.dtype,
                        )

                        init_weights[weight_name] = torch.cat(
                            (target_weight, zero), dim=1
                        )

                    # expand output channels
                    elif model_shape[0] > target_shape[0]:
                        diff = model_shape[0] - target_shape[0]
                        _, ic, h, w = target_shape

                        zero = torch.zeros(
                            (diff, ic, h, w),
                            dtype=target_weight.dtype,
                        )

                        init_weights[weight_name] = torch.cat(
                            (target_weight, zero), dim=0
                        )

                    else:
                        init_weights[weight_name] = model_weight.clone()

                # -------- Bias --------
                elif len(model_shape) == 1:
                    if model_shape[0] > target_shape[0]:
                        diff = model_shape[0] - target_shape[0]

                        zero = torch.zeros(
                            (diff,),
                            dtype=target_weight.dtype,
                        )

                        init_weights[weight_name] = torch.cat(
                            (target_weight, zero), dim=0
                        )
                    else:
                        init_weights[weight_name] = model_weight.clone()

        else:
            print(f"[New weight] {weight_name}")
            init_weights[weight_name] = scratch_weights[weight_name].clone()

    model.load_state_dict(init_weights, strict=True)
    return model


# =========================================================
# Trainer
# =========================================================
class Trainer:
    def __init__(self):
        self.sd_weights = (
            "D:\\kanyamahanga\\Datasets\\CMSRD\\checkpoints\\v2-1_512-ema-pruned.ckpt"
        )
        self.output = "./scripts/init_weight-ds.pt"

    def build_model(self):
        self.denoise_net = Unet(
            image_size=16,
            in_channels=4,
            out_channels=4,
            model_channels=64,
            attention_resolutions=[4, 2, 1],
            num_res_blocks=2,
            channel_mult=[1, 2, 4, 4],
            num_head_channels=32,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=1,
            context_dim=1024,
        )

        first_stage_config = {
            "embed_dim": 4,
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 4,
            "out_ch": 4,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }

        cond_stage_config = {}

        self.cond_net = TConvFormer(
            input_size=(hparams["sat_patch_size"], hparams["sat_patch_size"]),
            stem_channels=64,
            block_channels=hparams["block_channels"][:2],
            block_layers=hparams["block_layers"][1:],
            head_dim=32,
            stochastic_depth_prob=0.2,
            partition_size=4,
        )

        model = LatentDiffusion(
            denoise_net=self.denoise_net,
            cond_net=self.cond_net,
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            timesteps=hparams["timesteps"],
        )

        return model

    def load(self):
        model = self.build_model()

        sd_weights = load_weight(self.sd_weights)
        scratch_weights = model.state_dict()

        # ================= ANALYSIS =================
        stats = analyze_weight_mapping(scratch_weights, sd_weights)

        print("\n===== ANALYSIS =====")
        print("Exact match:", len(stats["exact_match"]))
        print("Same shape diff:", len(stats["same_shape_diff"]))
        print("Shape mismatch:", len(stats["shape_mismatch"]))
        print("Missing:", len(stats["missing"]))

        # Top similarities
        sorted_sim = sorted(
            stats["similarity"].items(),
            key=lambda x: x[1] if x[1] is not None else -1,
            reverse=True,
        )

        print("\nTop similar layers:")
        for k, v in sorted_sim[:10]:
            print(f"{k}: {v:.4f}")

        # ================= LOAD =================
        model = initialize_weights(model, sd_weights)

        torch.save(model.state_dict(), self.output)
        print("\nDone.")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    set_hparams()
    config = read_config(hparams["config_file"])

    trainer = Trainer()
    trainer.load()
