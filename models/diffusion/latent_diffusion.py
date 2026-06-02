import torch
import einops
import pathlib
from typing import Union
import requests
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Literal
from trainer import Trainer
import torch.nn.functional as F
from contextlib import contextmanager
from functools import partial
from utils.hparams import hparams
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from models.diffusion.utils import DDIMSampler
from skimage.exposure import match_histograms
from utils.utils import revert_padding, linear_transform_6b, assert_tensor_validity
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from models.autoencoder.autoencoder import DiagonalGaussianDistribution, AutoencoderKL

from models.diffusion.utils import (
    LitEma,
    count_params,
    default,
    disabled_train,
    exists,
    extract_into_tensor,
    make_beta_schedule,
    make_convolutional_sample,
)

from losses.srdiff_loss import (
    pixel_wise_closest_sr_sits_aer_loss,
    grad_pixel_wise_closest_sr_sits_aer_loss,
    temp_gradient_magnitude_consistency_loss,
    gray_value_consistency_loss,
)
from losses.focal_smooth import FocalLossWithSmoothing
from models.diffusion.modules.encoders.modules import FrozenOpenCLIPEmbedder

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


class DDPM(nn.Module):
    """This class implements the classic DDPM (Diffusion Models) with Gaussian diffusion in image space.

    Args:
        denoise_net (dict): A UNet network for the denoising.
        cond_net (dict): A temporal encoder network for encoding LR-SITS.
        timesteps (int): The number of diffusion timesteps to use during training(default to 500 DDPM).
        beta_schedule (str): The type of beta schedule to use (linear, cosine, or fixed).
        use_ema (bool): Whether to use exponential moving averages (EMAs) of the model weights during training.
        first_stage_key (str): The key to use for the first stage of the model (either "image" or "noise").
        linear_start (float): The starting value for the linear beta schedule.
        linear_end (float): The ending value for the linear beta schedule.
        cosine_s (float): The scaling factor for the cosine beta schedule.
        given_betas (list): A list of beta values to use for the fixed beta schedule.
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.
        conditioning_key (str): The type of conditioning to use (None, 'concat', 'crossattn', 'hybrid', or 'adm').
        parameterization (str): The type of parameterization to use for the diffusion process (either "eps" or "x0").
        use_positional_encodings (bool): Whether to use positional encodings for the input.

    Methods:
        register_schedule: Registers the schedule for the betas and alphas.
        get_input: Gets the input from the DataLoader and rearranges it.
        decode_first_stage: Decodes the first stage of the model.
        ema_scope: Switches to EMA weights during training.

    Attributes:
        parameterization (str): The type of parameterization used for the diffusion process.
        cond_stage_model (None): The conditioning stage model (not used in this implementation).
        first_stage_key (str): The key used for the first stage of the model.
        use_positional_encodings (bool): Whether positional encodings are used for the input.
        model (DiffusionWrapper): The diffusion model.
        use_ema (bool): Whether EMAs of the model weights are used during training.
        model_ema (LitEma): The EMA of the model weights.
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.

    Example:
        >>> model = DDPM(
                denoise_net, timesteps=1000, beta_schedule='linear',
                use_ema=True, first_stage_key='image'
            )
    """

    def __init__(
        self,
        denoise_net,
        cond_net,
        timesteps: int = 1000,
        loss_type: str = "l2",
        beta_schedule: str = "linear",
        use_ema: bool = True,
        first_stage_key: str = "image",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        given_betas: Optional[List[float]] = None,
        original_elbo_weight: float = 0.0,
        v_posterior: float = 0.0,
        l_simple_weight: float = 1.0,
        conditioning_key: Optional[str] = None,
        parameterization: str = "eps",
        use_positional_encodings: bool = False,
        learn_logvar: bool = False,
        logvar_init: float = 0.0,
    ) -> None:
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization

        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_net = cond_net
        self.cond_stage_model = None
        self.first_stage_key = first_stage_key
        self.use_positional_encodings = use_positional_encodings
        self.denoise_net = denoise_net
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.learn_logvar = learn_logvar

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.logvar = torch.full(fill_value=logvar_init, size=(self.timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        count_params(self.denoise_net, verbose=True)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.denoise_net)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.v_posterior = v_posterior

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        # # if self.frozen_diff:
        # self.denoise_net.eval()
        # # self.model.train = disabled_train
        # for name, param in self.denoise_net.named_parameters():
        #     # if "attn" not in name:
        #     #     param.requires_grad = False
        #     # else:
        #     param.requires_grad = False

    def register_schedule(
        self,
        given_betas: Optional[List[float]] = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        """
        Registers the schedule for the betas and alphas.

        Args:
            given_betas (list, optional): A list of beta values to use for the fixed beta schedule.
                Defaults to None.
            beta_schedule (str, optional): The type of beta schedule to use (linear, cosine, or fixed).
                Defaults to "linear".
            timesteps (int, optional): The number of diffusion timesteps to use. Defaults to 1000.
            linear_start (float, optional): The starting value for the linear beta schedule. Defaults to 1e-4.
            linear_end (float, optional): The ending value for the linear beta schedule. Defaults to 2e-2.
            cosine_s (float, optional): The scaling factor for the cosine beta schedule. Defaults to 8e-3.
        """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":  # epselon - noise
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_input(self, batch: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        """
        Gets the input from the DataLoader and rearranges it.

        Args:
            batch (Dict[str, torch.Tensor]): The batch of data from the DataLoader.
            k (str): The key for the input tensor in the batch.

        Returns:
            torch.Tensor: The input tensor, rearranged and converted to float.
        """

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @contextmanager
    def ema_scope(self, context: Optional[str] = None) -> Generator[None, None, None]:
        """
        A context manager that switches to EMA weights during training.

        What are EMA weights?

        EMA weights are a smoothed version of the model's parameters. Instead of using just the latest update, they are computed as:
        EMA = α · previous_EMA + (1-α) · current_weights
        This makes EMA weights:
            Less noisy
            More stable
            Often better for evaluation or inference

        What “switches to EMA weights during training” implies

        During training, the system:

        1. Keeps two sets of weights
            The normal (actively updated) weights
            The EMA (smoothed) weights

        2. Switches to the EMA weights temporarily for things like:
            Validation
            Computing metrics
            Teacher guidance (in distillation or consistency training)

        3. Then switches back to the normal weights to continue training updates

        Why this is done

        Using EMA weights can:
            Improve validation accuracy
            Give more reliable metrics
            Stabilize training, especially in:
            Diffusion models
            Self-supervised learning
            GANs
            Large neural networks
        Args:
            context (Optional[str]): A string to print when switching to and from EMA weights.

        Yields:
            None
        """
        if self.use_ema:
            self.model_ema.store(self.denoise_net.parameters())
            self.model_ema.copy_to(self.denoise_net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.denoise_net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the first stage of the model.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded output tensor.
        """
        # z = 1.0 / z.flatten().std() * z
        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                output_list = [
                    self.first_stage_model.decode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded

            else:
                return self.first_stage_model.decode(z)

        else:
            return self.first_stage_model.decode(z)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Samples from the posterior distribution at the given timestep.

        Args:
            x_start (torch.Tensor): The starting tensor.
            t (int): The timestep.
            noise (Optional[torch.Tensor]): The noise tensor.

        Returns:
            torch.Tensor: The sampled tensor.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # print("noise:", noise.shape)
        # print("x_start:", x_start.shape)

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )


class LatentDiffusion(DDPM):
    """
    LatentDiffusion is a class that extends the DDPM class and implements a diffusion
    model with a latent variable. The model consists of two stages: a first stage that
    encodes the input tensor into a latent tensor, and a second stage that decodes the
    latent tensor into the output tensor. The model also has a conditional stage that
    takes a conditioning tensor as input and produces a learned conditioning tensor
    that is used to condition the first and second stages of the model. The class
    provides methods for encoding and decoding tensors, computing the output tensor
    and loss, and sampling from the distribution at a given latent tensor and timestep.
    The class also provides methods for registering and applying schedules, and for
    getting and setting the scale factor and conditioning key.

    Methods:
        register_schedule(self, schedule: Schedule) -> None: Registers the given schedule
            with the model.
        make_cond_schedule(self, schedule: Schedule) -> Schedule: Returns a new schedule
            with the given schedule applied to the conditional stage of the model.
        encode_first_stage(self, x: torch.Tensor, t: int) -> torch.Tensor: Encodes the given
            input tensor with the first stage of the model for the given timestep.
        get_first_stage_encoding(self, x: torch.Tensor, t: int) -> torch.Tensor: Returns the
            encoding of the given input tensor with the first stage of the model for the
            given timestep.
        get_learned_conditioning(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
            Returns the learned conditioning tensor for the given input
            tensor, timestep, and conditioning tensor.
        get_input(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
            Returns the input tensor for the given input tensor, timestep, and
            conditioning tensor.
        compute(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Computes the output tensor and loss for the given input tensor,
            timestep, and conditioning tensor.
        apply_model(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor: Applies
            the model to the given input tensor, timestep, and conditioning tensor.
        get_fold_unfold(self, ks: int, stride: int, vqf: int) -> Tuple[Callable, Callable]: Returns the fold
            and unfold functions for the given kernel size, stride, and vector quantization factor.
        forward(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor: Computes the
            output tensor for the given input tensor, timestep, and conditioning tensor.
        q_sample(self, z: torch.Tensor, t: int, eps: Optional[torch.Tensor] = None) -> torch.Tensor: Samples
            from the distribution at the given latent tensor and timestep.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Union[str, Dict[str, Any]],
        num_timesteps_cond: Optional[int] = None,
        cond_stage_key: str = "image",
        cond_stage_trainable: bool = False,
        concat_mode: bool = True,
        cond_stage_forward: Optional[Callable] = None,
        conditioning_key: Optional[str] = None,
        scale_factor: float = 1.0,
        scale_by_std: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the LatentDiffusion model.

        Args:
            first_stage_config (Dict[str, Any]): The configuration for the first stage of the model.
            cond_stage_config (Union[str, Dict[str, Any]]): The configuration for the conditional stage of the model.
            num_timesteps_cond (Optional[int]): The number of timesteps for the conditional stage of the model.
            cond_stage_key (str): The key for the conditional stage of the model.
            cond_stage_trainable (bool): Whether the conditional stage of the model is trainable.
            concat_mode (bool): Whether to use concatenation or cross-attention for the conditioning.
            cond_stage_forward (Optional[Callable]): A function to apply to the output of the conditional
            stage of the model.
            conditioning_key (Optional[str]): The key for the conditioning.
            scale_factor (float): The scale factor for the input tensor.
            scale_by_std (bool): Whether to scale the input tensor by its standard deviation.
            *args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        self.linear_transform = linear_transform_6b

        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))

        self.cond_stage_forward = cond_stage_forward

        # Load pretrained VAE
        self.first_stage_model = AutoencoderKL.from_pretrained(
            hparams["diff_sat_weights"],
            subfolder="vae",
            sample_size=64,
            revision=None,
            low_cpu_mem_usage=False,
            device_map=None,
            ignore_mismatched_sizes=True,
        )

        # =========================
        # 1️⃣ Adapt Encoder for 4 channels
        # =========================
        old_conv = self.first_stage_model.encoder.conv_in
        new_conv = nn.Conv2d(
            in_channels=4,  # new input channels
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            # Copy pretrained weights for the first 3 channels
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialize the 4th channel as the mean of RGB channels (or use kaiming_normal)
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        # Replace encoder conv_in
        self.first_stage_model.encoder.conv_in = new_conv

        # =========================
        # 2️⃣ Adapt Decoder for 4 channels
        # =========================
        # Typically decoder output is conv_out

        # Decoder conv_out typically maps latent features -> image channels
        decoder_conv = self.first_stage_model.decoder.conv_out
        new_decoder_conv = nn.Conv2d(
            in_channels=decoder_conv.in_channels,  # keep latent channels
            out_channels=4,  # new image channels
            kernel_size=decoder_conv.kernel_size,
            stride=decoder_conv.stride,
            padding=decoder_conv.padding,
            bias=decoder_conv.bias is not None,
        )

        with torch.no_grad():
            # Copy pretrained weights for first 3 channels of output
            new_decoder_conv.weight[:3, :, :, :] = (
                decoder_conv.weight
            )  # output channels dimension
            # Initialize 4th channel randomly
            nn.init.kaiming_normal_(new_decoder_conv.weight[3:, :, :, :])
            if decoder_conv.bias is not None:
                new_decoder_conv.bias[:3] = decoder_conv.bias
                nn.init.zeros_(new_decoder_conv.bias[3:])

        # Replace decoder conv_out
        self.first_stage_model.decoder.conv_out = new_decoder_conv

        # =========================
        # 3️⃣ Freeze VAE except first conv layer (optional)
        # =========================
        # self.first_stage_model.requires_grad_(False)
        # self.first_stage_model.encoder.conv_in.requires_grad_(True)
        # self.first_stage_model.decoder.conv_out.requires_grad_(True)

        print("✅ VAE encoder and decoder updated for 4 channels")

        # ⚠️ Important detail
        # keep it frozen (still works surprisingly well)
        # ✔ or:
        # unfreeze ONLY first layer:

        # self.first_stage_model.requires_grad_(False)
        # self.first_stage_model.encoder.conv_in.requires_grad_(True)

        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        # Setup the CLIP model - use pretrained weights
        self.cond_stage_model = FrozenOpenCLIPEmbedder(hparams)
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for name, param in self.cond_stage_model.named_parameters():
            param.requires_grad = False

        # Similar to https://github.com/wwangcece/SGDM/blob/main/model/refsr_ldm_adapter_real.py#L96
        # All the weights of the U-Net denoiser are frozen except attention-based layers

        self.denoise_net.eval()
        # self.model.train = disabled_train
        for name, param in self.denoise_net.named_parameters():
            if "attn" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def register_schedule(
        self,
        given_betas: Optional[Union[float, torch.Tensor]] = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        """
        Registers the given schedule with the model.

        Args:
            given_betas (Optional[Union[float, torch.Tensor]]): The betas for the schedule.
            beta_schedule (str): The type of beta schedule to use.
            timesteps (int): The number of timesteps for the schedule.
            linear_start (float): The start value for the linear schedule.
            linear_end (float): The end value for the linear schedule.
            cosine_s (float): The scale factor for the cosine schedule.
        """
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def make_cond_schedule(self) -> None:
        """
        Shortens the schedule for the conditional stage of the model.
        """
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    # 2023-04-08
    def decode_first_stage_with_grad(
        self, z, predict_cids=False, force_not_quantize=False
    ):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        # z = 1.0 / z.flatten().std() * z
        z = 1.0 / self.scale_factor * z

        return self.first_stage_model.decode(z)

    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the given input tensor with the first stage of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded output tensor.
        """
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(
        self, encoder_posterior: Union[DiagonalGaussianDistribution, torch.Tensor]
    ) -> torch.Tensor:
        """
        Returns the encoding of the given input tensor with the first stage of the
        model for the given timestep.

        Args:
            encoder_posterior (Union[DiagonalGaussianDistribution, torch.Tensor]): The
                encoder posterior.

        Returns:
            torch.Tensor: The encoding of the input tensor.
        """
        if hasattr(encoder_posterior, "sample"):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        # return z.flatten().std() * z
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)

        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        with torch.no_grad():
            x, c = super().get_input(batch, *args, **kwargs)
        return x, dict(c_crossattn=[c])

    def apply_model(
        self,
        x_noisy: torch.Tensor,
        t: int,
        txt: Optional[torch.Tensor] = None,
        mtd: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        cond_hr: Optional[torch.Tensor] = None,
        return_ids: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Applies the model to the given noisy (latent) input tensor.

        Args:
            x_noisy (torch.Tensor): The noisy input tensor.
            t (int): The timestep.
            cond (Optional[torch.Tensor]): The conditioning tensor.
            cond_txt (Optional[torch.Tensor]): The conditioning text tensor.

            return_ids (bool): Whether to return the IDs of the diffusion process.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output tensor,
             and optionally the IDs of the
            diffusion process.
            returns a latent, which is then used for loss computation
            So the loss is computed directly on the latent space, not the decoded image.
        """

        # diffusion prediction
        # print("x_noisy", x_noisy.shape)
        # print("cond_txt", cond_txt.shape)
        # print("cond[0]", cond[0].shape)

        x_recon = self.denoise_net(
            x_noisy,
            t,
            encoder_hidden_states=txt,  # these are supposed to be text prompts
            class_labels=None,  # we can not use the text twice
            metadata=mtd,
            down_block_additional_residuals=cond,  # conditioning with LR images
            up_block_additional_residuals=cond_hr,  # conditioning with HR images
        )

        if isinstance(x_recon, tuple) and not return_ids:
            # print("x_recon[0]", x_recon[0].shape)
            return x_recon[0]
        else:
            # print("x_recon", x_recon.shape) # torch.Size([2, 4, 16, 16])
            return x_recon

    def get_fold_unfold(
        self, x: torch.Tensor, kernel_size: int, stride: int, uf: int = 1, df: int = 1
    ) -> Tuple[nn.Conv2d, nn.ConvTranspose2d]:
        """
        Returns the fold and unfold convolutional layers for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            kernel_size (int): The kernel size.
            stride (int): The stride.
            uf (int): The unfold factor.
            df (int): The fold factor.

        Returns:
            Tuple[nn.Conv2d, nn.ConvTranspose2d]: The fold and unfold convolutional layers.
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h * uf, w * uf
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )

        elif df > 1 and uf == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h // df, w // df
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx)
            )

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    def closest_lr_sits_aer(self, img_lr, closest_indices):
        B, T, C, H, W = img_lr.shape
        closest_indices = closest_indices.to(torch.long)

        # Ensure closest_indices is a tensor
        if not torch.is_tensor(closest_indices):
            closest_indices = torch.tensor(closest_indices, device=img_lr.device)

        # Gather closest satellite images
        closest_sat_image = img_lr[
            torch.arange(B), closest_indices
        ]  # shape (B, C, H, W)
        return closest_sat_image

    def apply_cond_lr_encoder(self, img_lr, dates):
        # features are extracted at 1.6, 3.2, 6.4m GSD
        _, cond = self.cond_net(img_lr, dates)
        # cond = [cond * self.scale_factor for cond in cond] # scaling the cond features
        return cond

    def apply_cond_hr_encoder(self, img_hr):
        # features are extracted at , 1.6, 3.2, 6.4m GSD
        _, _, hr_2, hr_3, hr_4 = self.aer_net_enc(img_hr)
        cond_hr = [hr_2, hr_3, hr_4]
        # cond_hr = [cond_hr * self.scale_factor for cond_hr in cond_hr] # scaling the cond features
        return cond_hr

    def p_losses(
        self,
        x_start,
        t,
        txt,
        mtd,
        cond_lr,
        cond_hr,
        img_hr,
        img_lr,
        img_lr_up,
        closest_idx,
        noise=None,
    ):
        noise = default(noise, lambda: torch.randn_like(x_start))

        model_outputs = []
        sr_outputs = []

        for tps in range(cond_lr[0].shape[1]):
            # latent prediction from diffusion model
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_output = self.apply_model(
                x_noisy,
                t,
                txt,
                mtd,
                [feat[:, tps] for feat in cond_lr],
                cond_hr,
            )
            # model_output: torch.Size([2, 4, 64, 64])

            # recover clean latent - assuming parameterization == "eps"
            x0_pred = self.predict_start_from_noise(x_noisy, t, model_output)
            # x0_pred: torch.Size([2, 4, 64, 64]

            # print("x0_pred:", x0_pred.shape)
            # print("Min x0_pred:", x0_pred.min().item())
            # print("Max x0_pred:", x0_pred.max().item())
            # print()
            # generate SR image for classification
            sr_dec = self.decode_first_stage_with_grad(x0_pred)["sample"]
            # x_sr: torch.Size([2, 4, 64, 64]

            # print("sr_dec:", sr_dec.shape)
            # print("Min sr_dec:", sr_dec.min().item())
            # print("Max sr_dec:", sr_dec.max().item())
            # print()
            # x_sr = self.res2img(sr_dec, img_lr_up[:, tps, :, :])

            model_outputs.append(model_output)
            sr_outputs.append(sr_dec)

        model_outputs = torch.stack(model_outputs, 1)  # torch.Size([4, 6, 4, 16, 16])
        sr_outputs = torch.stack(sr_outputs, 1)  # torch.Size([4, 6, 4, 64, 64])

        if self.parameterization == "x0":  # predicted image
            target = x_start
        elif self.parameterization == "eps":  # predicted noise
            target = noise  # torch.Size([2, 4, 16, 16])
        else:
            raise NotImplementedError()

        # print("target:", target.shape)
        # print("model_outputs:", model_outputs.shape)
        # print("x:", x.shape)

        noise_pred = self.closest_lr_sits_aer(model_outputs, closest_idx)
        loss = self.get_loss(noise_pred, target)

        # auxiliary losses computations
        aux_loss = (
            hparams["px_loss_weight"]
            * pixel_wise_closest_sr_sits_aer_loss(sr_outputs, img_hr, closest_idx)
            + hparams["grad_px_loss_weight"]
            * grad_pixel_wise_closest_sr_sits_aer_loss(sr_outputs, img_hr, closest_idx)
            + hparams["temp_grad_mag_loss_weight"]
            * temp_gradient_magnitude_consistency_loss(sr_outputs)
            + hparams["gray_value_px_loss_weight"]
            * gray_value_consistency_loss(sr_outputs, img_lr)
        )
        final_loss = hparams["main_loss_weight"] * loss + aux_loss
        return final_loss, sr_outputs

    def forward(
        self,
        img: torch.Tensor,  # HR img_hr
        img_hr: torch.Tensor,  # downsampled img_hr
        img_lr: torch.Tensor,  # LR images
        img_lr_up,  # Upsampled LR images
        txt: list,
        mtd: torch.Tensor,
        dates: torch.Tensor,
        closest_idx: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor - HR image.
            img_lr (torch.Tensor): The input tensor - LR image for conditioning
            dates (torch.Tensor): The input tensor - acquisition dates

            closest_idx (torch.Tensor): closest idx to HR image acquisition tensor
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """

        # x_res = [
        #     self.img2res(img, img_lr_up[:, i]) for i in range(img_lr_up.shape[1])
        # ]
        # x_start = torch.stack(x_res, dim=1)

        # x_start_res = self.closest_lr_sits_aer(x_start, closest_idx)
        x = img[:, :4, :, :]
        encoder_posterior = self.encode_first_stage(x).latent_dist

        # from  DiffusionSat x: 512x512 - 20 cm
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        # z: 64x64 - 1.6 m
        txt = self.get_learned_conditioning(txt)
        cond_lr = self.apply_cond_lr_encoder(img_lr, dates)
        cond_hr = self.apply_cond_hr_encoder(img)

        t = torch.randint(
            0, self.num_timesteps, (z.shape[0],), device=self.device
        ).long()  # self.num_timesteps = 500

        loss, model_outputs = self.p_losses(
            z,
            t,
            txt,
            mtd,
            cond_lr,
            cond_hr,
            img_hr,
            img_lr,
            img_lr_up,
            closest_idx,
            *args,
            **kwargs,
        )
        losses = {"sr": loss}
        return losses, model_outputs

    def _prepare_model(
        self,
        eta: float = 1.0,
        custom_steps: int = 100,
        verbose: bool = False,
    ):
        # Create the DDIM sampler with uses 100 time steps during inference
        ddim = DDIMSampler(self)

        # make schedule to compute alphas and sigmas
        ddim.make_schedule(ddim_num_steps=custom_steps, ddim_eta=eta, verbose=verbose)

        # Create the vector with the timesteps
        timesteps = ddim.ddim_timesteps
        time_range = np.flip(timesteps)
        return ddim, time_range

    @torch.no_grad()
    def sample(
        self,
        img,
        img_hr,
        img_lr,  # low-resolution image used for conditioning
        img_lr_up,
        dates,
        txt,
        mtd,  # metadata for conditioning
        closest_idx,
        eta: float = 1.0,
        temperature: float = 1.0,
        custom_steps: int = 100,  # used for inference
        verbose: bool = False,
        histogram_matching: bool = True,
        save_iterations: bool = False,
    ):
        txt = self.get_learned_conditioning(txt)
        cond_lr = self.apply_cond_lr_encoder(img_lr, dates)
        cond_hr = self.apply_cond_hr_encoder(img)

        # Assert shape, size, dimensionality
        shape = img_hr.shape  # torch.Size([1, 4, 64, 64])
        latent_shape = (shape[0], shape[1], shape[2], shape[3])

        # ddim, latent and time_range
        ddim, time_range = self._prepare_model(
            eta=eta, custom_steps=custom_steps, verbose=verbose
        )

        iterator = tqdm(
            time_range, desc="DDIM Sampler", total=custom_steps, disable=True
        )
        # print("latent gen:", latent.shape)
        # print("custom_steps:", custom_steps)

        sr_images = []

        for j in range(img_lr.shape[1]):
            # Each timestep in the reverse diffusion process is conditioned
            # the corresponding temporal features from the LR image

            # Iterate over the timesteps
            if save_iterations:
                save_iters = []

            # Create the HR latent image
            latent = torch.randn(latent_shape, device=img_hr.device)

            for i, step in enumerate(iterator):
                outs = ddim.p_sample_ddim(
                    x=latent,
                    c=[feat[:, j] for feat in cond_lr],
                    txt=txt,
                    mtd=mtd,
                    cond_hr=cond_hr,
                    t=step,
                    index=custom_steps - i - 1,
                    use_original_steps=False,
                    temperature=temperature,
                )
                latent, _ = outs

                if save_iterations:
                    sr_dec = self.decode_first_stage(latent)["sample"]
                    # sr_dec = self.res2img(sr_dec, img_lr_up[:, j, :, :])
                    save_iters.append(sr_dec)

            if save_iterations:
                return save_iters

            sr_dec = self.decode_first_stage(latent)["sample"]
            # print("min sr_dec:", sr_dec.min())
            # print("max sr_dec:", sr_dec.max())
            # sr_dec = self.res2img(sr_dec, img_lr_up[:, j, :, :])
            sr_images.append(sr_dec)

        img_sr = torch.stack(sr_images, 1)  # torch.Size([2, 4, 4, 64, 64]
        sr_pred = self.closest_lr_sits_aer(img_sr, closest_idx)
        loss = self.get_loss(sr_pred, img_hr)

        # auxiliary losses computations
        aux_loss = (
            hparams["px_loss_weight"]
            * pixel_wise_closest_sr_sits_aer_loss(img_sr, img_hr, closest_idx)
            + hparams["grad_px_loss_weight"]
            * grad_pixel_wise_closest_sr_sits_aer_loss(img_sr, img_hr, closest_idx)
            + hparams["temp_grad_mag_loss_weight"]
            * temp_gradient_magnitude_consistency_loss(img_sr)
            + hparams["gray_value_px_loss_weight"]
            * gray_value_consistency_loss(img_sr, img_lr)
        )
        final_loss = hparams["main_loss_weight"] * loss + aux_loss
        return img_sr, final_loss

    def img2res(self, img, img_lr_up, clip_input=None):
        # Compute Residual Between HR and LR-upsampled image
        # Subtract the upsampled LR image from the HR image
        # to obtain the residual.

        # x: torch.Size([5, 3, 40, 40])
        # img_lr_up: torch.Size([5, 2, 3, 40, 40])
        img = img[:, :4, :, :]
        if clip_input is None:
            clip_input = hparams["clip_input"]
        if hparams["res"]:
            res = (img - img_lr_up) * hparams["res_rescale"]
            if clip_input:
                res = res.clamp(-1, 1)
        return res

    def res2img(self, res, img_lr_up, clip_input=None):
        res = res[:, :4, :, :]
        if clip_input is None:
            clip_input = hparams["clip_input"]
        if hparams["res"]:
            if clip_input:
                res = res.clamp(-1, 1)
            # rescales the residual image before adding it to the upsampled LR image
            # with a scaling factor defined by hparams['res_rescale'] for stability.
        img = res / hparams["res_rescale"] + img_lr_up
        return img
