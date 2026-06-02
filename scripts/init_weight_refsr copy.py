import sys

sys.path.append(".")
import torch
import importlib
from typing import Dict
from utils.hparams import hparams, set_hparams
from utils.utils_dataset import read_config
from utils.hparams import hparams
from models.denoiser.unet import Unet
from models.diffusion.latent_diffusion import LatentDiffusion
from models.encoders.t_convformer import TConvFormer


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


class Trainer:
    def __init__(self):
        self.sd_weights = (
            "D:\\kanyamahanga\\Datasets\\CMSRD\\checkpoints\\v2-1_512-ema-pruned.ckpt"
        )
        self.output = "./checkpoints/init_weight/init_weight-sd.pt"

    def load(self):
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        self.denoise_net = Unet(
            image_size=16,  # unused
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
        cond_stage_config = {
            "image_size": 64,
            "in_channels": 8,
            "model_channels": 160,
            "out_channels": 4,
            "num_res_blocks": 2,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 2, 2, 4],
            "num_head_channels": 32,
        }

        # 2. Cond Encoder with ONLY two encoding block layers
        self.cond_net = TConvFormer(
            input_size=(hparams["sat_patch_size"], hparams["sat_patch_size"]),
            stem_channels=64,
            block_channels=hparams["block_channels"][
                :2
            ],  # [128, 256, 512],  # [64, 128, 256, 512]
            block_layers=hparams["block_layers"][1:],  # [2, 2, 5],  # [2, 2, 5, 2]
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

        sd_weights = load_weight(self.sd_weights)

        scratch_weights = model.state_dict()

        print(
            " ------------------------ scratch_weights.keys() ------------------------------------ :",
            scratch_weights.keys(),
        )

        init_weights = {}
        for weight_name in scratch_weights.keys():
            # find target pretrained weights for this weight
            if weight_name.startswith("control_"):
                suffix = weight_name[len("control_") :]
                target_name = f"latent_diff.diffusion_{suffix}"
                target_model_weights = sd_weights
            else:
                target_name = weight_name
                target_model_weights = sd_weights

            # if target weight exist in pretrained model
            print(f"copy weights: {target_name} -> {weight_name}")
            # if target_name in target_model_weights and ('transformer_blocks' not in target_name):
            if target_name in target_model_weights:
                # get pretrained weight
                target_weight = target_model_weights[target_name]
                target_shape = target_weight.shape
                model_shape = scratch_weights[weight_name].shape
                # print("model_shape:", model_shape)
                # print("target_shape:", target_shape)

                # if pretrained weight has the same shape with model weight, we make a copy
                if model_shape == target_shape:
                    init_weights[weight_name] = target_weight.clone()
                # else we copy pretrained weight with additional channels initialized to zero
                else:
                    print("model_shape:", model_shape)
                    print("target_shape:", target_shape)
                    print("target_name:", target_name)

                    # -------- Conv weight (4D) --------
                    if len(model_shape) == 4:  # Conv weight (4D) layer
                        # Check for input channel mismatch
                        if model_shape[1] != target_shape[1]:  # Compare input channels
                            diff = model_shape[1] - target_shape[1]

                            if diff < 0:
                                raise ValueError("Cannot shrink pretrained weights")

                            # Ensure height (h) and width (w) match
                            oc, ic, h, w = target_shape

                            # Create a zero tensor for the new input channels with the same spatial dimensions (h, w)
                            zero_weight = torch.zeros(
                                (oc, diff, h, w),
                                device=target_weight.device,
                                dtype=target_weight.dtype,
                            )

                            print("target_weight:", target_weight.shape)
                            print("zero_weight:", zero_weight.shape)

                            # Concatenate zero-weight tensor along input channel dimension
                            init_weights[weight_name] = torch.cat(
                                (target_weight.clone(), zero_weight), dim=1
                            )

                            print(
                                f"Expanded weight: {weight_name}, +{diff} input channels"
                            )

                        # Check for output channel mismatch
                        elif (
                            model_shape[0] != target_shape[0]
                        ):  # Compare output channels
                            diff = model_shape[0] - target_shape[0]

                            if diff < 0:
                                raise ValueError("Cannot shrink pretrained weights")

                            # Ensure height (h) and width (w) match
                            _, ic, h, w = target_shape

                            print("target_weight:", target_weight.shape)
                            print("zero_weight:", zero_weight.shape)

                            # Create a zero tensor for the new output channels with the same spatial dimensions (h, w)
                            zero_weight = torch.zeros(
                                (diff, ic, h, w),
                                device=target_weight.device,
                                dtype=target_weight.dtype,
                            )

                            # Concatenate zero-weight tensor along output channel dimension
                            init_weights[weight_name] = torch.cat(
                                (target_weight.clone(), zero_weight), dim=0
                            )

                            print(
                                f"Expanded weight: {weight_name}, +{diff} output channels"
                            )

                        else:
                            # No mismatch, just clone the target weight
                            init_weights[weight_name] = target_weight.clone()

                    # -------- Bias (1D) --------
                    elif len(model_shape) == 1:
                        if model_shape[0] != target_shape[0]:
                            diff = model_shape[0] - target_shape[0]

                            if diff < 0:
                                raise ValueError("Cannot shrink bias")

                            zero_bias = torch.zeros(
                                (diff,),
                                device=target_weight.device,
                                dtype=target_weight.dtype,
                            )

                            init_weights[weight_name] = torch.cat(
                                (target_weight.clone(), zero_bias), dim=0
                            )

                            print(f"Expanded bias: {weight_name}, +{diff}")

                        else:
                            init_weights[weight_name] = target_weight

            else:
                init_weights[weight_name] = scratch_weights[weight_name].clone()
                print(f"These weights are newly added: {weight_name}")

        model.load_state_dict(init_weights, strict=True)
        torch.save(model.state_dict(), self.output)
        print("Done.")


if __name__ == "__main__":
    set_hparams()
    config = read_config(hparams["config_file"])
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    trainer = Trainer()
    trainer.load()


#  python scripts/init_weight_refsr.py --config configs_loc/diffsr_maxvit_ltae.yaml --config_file flair-config.yml --exp_name misr/srdiff_maxvit_ltae_ckpt --reset
