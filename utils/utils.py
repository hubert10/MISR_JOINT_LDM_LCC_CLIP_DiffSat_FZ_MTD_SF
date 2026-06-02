import glob, sys
import os, re
import lpips
import torch
import subprocess
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from .matlab_resize import imresize
import torchvision.models as models
from torchvision.models import alexnet
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.classification import MulticlassJaccardIndex
import torch
from einops import rearrange

sys.path.insert(0, "../")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def tensors_to_np(tensors):
    if isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = tensors_to_np(v)
        new_np = v
    else:
        raise Exception(f"tensors_to_np does not support type {type(tensors)}.")
    return new_np


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, "cuda", None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, "to", None)):
        return batch.to(torch.device("cuda", gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


import torch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_infos = get_all_ckpts(
        work_dir, steps
    )  # returns list of (path, step, val_miou)

    if len(ckpt_infos) > 0:
        last_ckpt_path = ckpt_infos[0][0]  # extract path from tuple
        checkpoint = torch.load(last_ckpt_path, map_location="cpu", weights_only=True)
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f"{work_dir}/model_ckpt_steps_*.ckpt"
    else:
        ckpt_path_pattern = f"{work_dir}/model_ckpt_steps_{steps}*.ckpt"

    ckpts = glob.glob(ckpt_path_pattern)
    ckpt_info = []

    for ckpt in ckpts:
        step_match = re.search(r"steps_(\d+)", ckpt)
        miou_match = re.search(r"val_miou_(\d+\.\d+)", ckpt)

        if step_match:
            step = int(step_match.group(1))
        else:
            step = -1  # fallback if not found

        if miou_match:
            val_miou = float(miou_match.group(1))
        else:
            val_miou = None

        ckpt_info.append((ckpt, step, val_miou))

    # Sort by step descending
    return sorted(ckpt_info, key=lambda x: -x[1])


def load_checkpoint(model, optimizer, work_dir):
    print("work_dir where to load weights:", work_dir)
    checkpoint, _ = get_last_checkpoint(work_dir)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"]["model"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        training_step = checkpoint["global_step"]
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        model.to(device)
    return training_step


def save_checkpoint(
    model, optimizer, work_dir, global_step, num_ckpt_keep, val_miou=None
):
    val_miou_str = f"_miou_{val_miou:.2f}" if val_miou is not None else ""
    ckpt_path = f"{work_dir}/model_ckpt_steps_{global_step}{val_miou_str}.ckpt"
    print(f"Step@{global_step}: saving model to {ckpt_path}")
    checkpoint = {"global_step": global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint["optimizer_states"] = optimizer_states
    checkpoint["state_dict"] = {"model": model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        print("old_ckpt:", old_ckpt)
        remove_file(old_ckpt)
        print(f"Delete ckpt: {os.path.basename(old_ckpt)}")


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def plot_img(img):
    img = img.data.cpu().numpy()
    return np.clip(img, 0, 1)


def load_ckpt(cur_model, ckpt_base_dir, model_name="model", force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location="cpu")
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir)

    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if len([k for k in state_dict.keys() if "." in k]) > 0:
            state_dict = {
                k[len(model_name) + 1 :]: v
                for k, v in state_dict.items()
                if k.startswith(f"{model_name}.")
            }
        else:
            state_dict = state_dict[model_name]
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)


class Measure:
    def __init__(self, net="alex"):
        self.model = lpips.LPIPS(net=net)

    def measure(self, imgA, imgB, img_lr, preds=None, labels=None):
        """
        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]

        Returns: dict of metrics
        """

        if isinstance(imgA, torch.Tensor):
            imgA = (
                np.round((imgA.cpu().numpy() + 1) * 127.5)
                .clip(min=0, max=255)
                .astype(np.int32)
            )
            imgB = (
                np.round((imgB.cpu().numpy() + 1) * 127.5)
                .clip(min=0, max=255)
                .astype(np.int32)
            )
            img_lr = (
                np.round((img_lr.cpu().numpy() + 1) * 127.5)
                .clip(min=0, max=255)
                .astype(np.int32)
            )
        imgA = imgA.transpose(1, 2, 0)
        imgB = imgB.transpose(1, 2, 0)
        psnr = self.psnr(imgA, imgB)
        ssim = self.ssim(imgA, imgB)
        lpips = self.lpips(imgA, imgB)
        mae = self.mae(imgA, imgB)
        mse = self.mse(imgA, imgB)
        shift_mae = self.shift_l1_loss(imgA, imgB)
        miou = self.miou(preds, labels) if preds is not None else 0.0

        res = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "mae": mae,
            "mse": mse,
            "shift_mae": shift_mae,
            "miou": miou,
        }
        return {k: float(v) for k, v in res.items()}

    def miou(self, x0_pred_logits, labels):
        # Assume `num_classes` is the number of semantic classes in your land cover labels
        miou_metric = MulticlassJaccardIndex(num_classes=13, average="macro")
        miou_metric = miou_metric.to(device)
        # Get predicted class map
        # pred_labels = torch.argmax(x0_pred_logits, dim=1)  # [B, H, W]
        # Compute mIoU
        # print("x0_pred_logits:", x0_pred_logits.shape)
        # print("labels:", labels.shape)

        miou = miou_metric(x0_pred_logits, labels)  # Scalar tensor
        miou = miou.item()  # convert from tensor to float if needed
        return miou

    def lpips(self, imgA, imgB, model=None):
        # compute LPIPS only on RGB channels
        imgA = imgA[:, :, :3]
        imgB = imgB[:, :, :3]
        device = next(self.model.parameters()).device
        tA = t(imgA).to(device)
        tB = t(imgB).to(device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        score, diff = ssim(imgA, imgB, full=True, channel_axis=-1, data_range=255)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)

    def mae(self, imgA, imgB):
        return np.abs(imgA - imgB).mean()

    def mse(self, imgA, imgB):
        return ((imgA.astype(np.int32) - imgB.astype(np.int32)) ** 2).mean()

    def shift_l1_loss(self, imgA, imgB, border=3):
        """
        Modified mae to take into account pixel shifts
        """
        y_true = imgB.astype(np.int32)
        y_pred = imgA.astype(np.int32)
        max_pixels_shifts = 2 * border
        size_image = y_true.shape[0]
        patch_pred = y_pred[
            border : size_image - border, border : size_image - border, :
        ]

        X = []
        for i in range(max_pixels_shifts + 1):
            for j in range(max_pixels_shifts + 1):
                patch_true = y_true[
                    i : i + (size_image - max_pixels_shifts),
                    j : j + (size_image - max_pixels_shifts),
                    :,
                ]
                l1_loss = np.mean(np.abs(patch_true - patch_pred))
                X.append(l1_loss)

        min_l1 = min(X)

        return min_l1


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def im_resize(batch, sr_factor):
    s = torch.tensor(batch[0][0].shape) * sr_factor
    img_lr_up = []
    iter_ = iter(batch)
    if batch.shape[0] == 1:
        s = np.squeeze(batch).shape[-1]
        im_up = cv.resize(
            np.array(np.transpose(np.squeeze(batch))),
            dsize=(s * sr_factor, s * sr_factor),
            interpolation=cv.INTER_CUBIC,
        )  # np.transpose(imresize(np.transpose(np.squeeze(batch)), sr_factor))
        return np.array(im_up)[None, :, :, :]

    for i in range(len(batch)):
        im_lr = next(iter_)
        s = im_lr.shape[-1]

        im_up = cv.resize(
            np.array(np.transpose(im_lr)),
            dsize=(s * sr_factor, s * sr_factor),
            interpolation=cv.INTER_CUBIC,
        )
        img_lr_up.append(np.transpose(im_up))
    return np.array(img_lr_up)


def range_image(image):
    # image in [-1,1]
    return image * 2 - 1


def multi_im_resize(batch, sr_factor):
    s = torch.tensor(batch[0][0].shape) * sr_factor
    img_lr_up = []
    iter_ = iter(batch)
    if batch.shape[0] == 1:
        s = np.squeeze(batch).shape[-1]
        im_up = cv.resize(
            np.array(np.transpose(np.squeeze(batch))),
            dsize=(s * sr_factor, s * sr_factor),
            interpolation=cv.INTER_CUBIC,
        )  # np.transpose(imresize(np.transpose(np.squeeze(batch)), sr_factor))
        return np.array(im_up)[None, :, :, :]

    for i in range(len(batch)):
        im_lr = next(iter_)
        s = im_lr.shape[-1]

        im_up = cv.resize(
            np.array(np.transpose(im_lr)),
            dsize=(s * sr_factor, s * sr_factor),
            interpolation=cv.INTER_CUBIC,
        )
        img_lr_up.append(np.transpose(im_up))
    return np.array(img_lr_up)


import torch
from einops import rearrange


def linear_transform_4b(t_input, stage="norm"):
    assert stage in ["norm", "denorm"]
    # get the shape of the tensor
    shape = t_input.shape

    # if 5 d tensor, norm/denorm individually
    if len(shape) == 5:
        stack = []
        for batch in t_input:
            stack2 = []
            for i in range(0, t_input.size(1), 4):
                slice_tensor = batch[i : i + 4, :, :, :]
                slice_denorm = linear_transform_4b(slice_tensor, stage=stage)
                stack2.append(slice_denorm)
            stack2 = torch.stack(stack2)
            stack2 = stack2.reshape(shape[1], shape[2], shape[3], shape[4])
            stack.append(stack2)
        stack = torch.stack(stack)
        return stack

    # here only if len(shape) == 4
    squeeze_needed = False
    if len(shape) == 3:
        squeeze_needed = True
        t_input = t_input.unsqueeze(0)
        shape = t_input.shape

    assert (
        len(shape) == 4 or len(shape) == 5
    ), "Input tensor must have 4 dimensions (B,C,H,W) - or 5D for MISR"
    transpose_needed = False
    if shape[-1] > shape[1]:
        transpose_needed = True
        t_input = rearrange(t_input, "b c h w -> b w h c")

    # define constants
    rgb_c = 3.0
    nir_c = 5.0

    # iterate over batches
    return_ls = []
    for t in t_input:
        if stage == "norm":
            # divide according to conventions
            t[:, :, 0] = t[:, :, 0] * (10.0 / rgb_c)  # R
            t[:, :, 1] = t[:, :, 1] * (10.0 / rgb_c)  # G
            t[:, :, 2] = t[:, :, 2] * (10.0 / rgb_c)  # B
            t[:, :, 3] = t[:, :, 3] * (10.0 / nir_c)  # NIR
            # clamp to get rif of outlier pixels
            t = t.clamp(0, 1)
            # bring to -1..+1
            t = (t * 2) - 1
        if stage == "denorm":
            # bring to 0..1
            t = (t + 1) / 2
            # divide according to conventions
            t[:, :, 0] = t[:, :, 0] * (rgb_c / 10.0)  # R
            t[:, :, 1] = t[:, :, 1] * (rgb_c / 10.0)  # G
            t[:, :, 2] = t[:, :, 2] * (rgb_c / 10.0)  # B
            t[:, :, 3] = t[:, :, 3] * (nir_c / 10.0)  # NIR
            # clamp to get rif of outlier pixels
            t = t.clamp(0, 1)

        # append result to list
        return_ls.append(t)

    # after loop, stack image
    t_output = torch.stack(return_ls)
    # print("stacked",t_output.shape)

    if transpose_needed == True:
        t_output = rearrange(t_output, "b w h c -> b c h w")
    if squeeze_needed:
        t_output = t_output.squeeze(0)

    return t_output


def linear_transform_6b(t_input, stage="norm"):
    # iterate over batches
    assert stage in ["norm", "denorm"]
    bands_c = 5.0
    return_ls = []
    clamp = False
    for t in t_input:
        if stage == "norm":
            # divide according to conventions
            t[:, :, 0] = t[:, :, 0] * (10.0 / bands_c)
            t[:, :, 1] = t[:, :, 1] * (10.0 / bands_c)
            t[:, :, 2] = t[:, :, 2] * (10.0 / bands_c)
            t[:, :, 3] = t[:, :, 3] * (10.0 / bands_c)
            t[:, :, 4] = t[:, :, 4] * (10.0 / bands_c)
            t[:, :, 5] = t[:, :, 5] * (10.0 / bands_c)
            # clamp to get rif of outlier pixels
            if clamp:
                t = t.clamp(0, 1)
            # bring to -1..+1
            t = (t * 2) - 1
        if stage == "denorm":
            # bring to 0..1
            t = (t + 1) / 2
            # divide according to conventions
            t[:, :, 0] = t[:, :, 0] * (bands_c / 10.0)
            t[:, :, 1] = t[:, :, 1] * (bands_c / 10.0)
            t[:, :, 2] = t[:, :, 2] * (bands_c / 10.0)
            t[:, :, 3] = t[:, :, 3] * (bands_c / 10.0)
            t[:, :, 4] = t[:, :, 4] * (bands_c / 10.0)
            t[:, :, 5] = t[:, :, 5] * (bands_c / 10.0)
            # clamp to get rif of outlier pixels
            if clamp:
                t = t.clamp(0, 1)

        # append result to list
        return_ls.append(t)

    # after loop, stack image
    t_output = torch.stack(return_ls)

    return t_output


def assert_tensor_validity(tensor):

    # ASSERT BATCH DIMENSION
    # if unbatched, add batch dimension
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    # ASSERT BxCxHxW ORDER
    # Check the size of the input tensor
    if tensor.shape[-1] < 10:
        tensor = rearrange(tensor, "b w h c -> b c h w")

    height, width = tensor.shape[-2], tensor.shape[-1]
    # Calculate how much padding is needed for height and width
    if height < 16 or width < 16:
        pad_height = max(0, 16 - height)  # Amount to pad on height
        pad_width = max(0, 16 - width)  # Amount to pad on width

        # Padding for height and width needs to be added to both sides of the dimension
        # The pad has the format (left, right, top, bottom)
        padding = (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        )
        padding = padding

        # Apply symmetric padding
        tensor = torch.nn.functional.pad(tensor, padding, mode="reflect")

    else:  # save padding with 0s
        padding = (0, 0, 0, 0)
        padding = padding

    return tensor, padding


def revert_padding(tensor, padding):
    left, right, top, bottom = padding
    # account for 4x upsampling Factor
    left, right, top, bottom = left * 4, right * 4, top * 4, bottom * 4
    # Calculate the indices to slice from the padded tensor
    start_height = top
    end_height = tensor.size(-2) - bottom
    start_width = left
    end_width = tensor.size(-1) - right

    # Slice the tensor to remove padding
    unpadded_tensor = tensor[:, :, start_height:end_height, start_width:end_width]
    return unpadded_tensor
