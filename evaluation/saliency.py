# Copyright 2022 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm
from scipy import ndimage

from evaluation.metrics.average_meter import AverageMeter
from evaluation.metrics.f_measure import FMeasure
from evaluation.metrics.iou import compute_iou
from evaluation.metrics.mae import compute_mae
from evaluation.metrics.pixel_acc import compute_pixel_accuracy
from evaluation.metrics.s_measure import SMeasure

from misc import batch_apply_bilateral_solver


@torch.no_grad()
def write_metric_tf(writer, metrics, n_iter=-1, name=""):
    writer.add_scalar(
        f"Validation/{name}iou_pred",
        metrics["ious"].avg,
        n_iter,
    )
    writer.add_scalar(
        f"Validation/{name}acc_pred",
        metrics["pixel_accs"].avg,
        n_iter,
    )
    writer.add_scalar(
        f"Validation/{name}f_max",
        metrics["f_maxs"].avg,
        n_iter,
    )


@torch.no_grad()
def eval_batch(batch_gt_masks, batch_pred_masks, metrics_res={}, reset=False):
    """
    Evaluation code adapted from SelfMask: https://github.com/NoelShin/selfmask
    """

    f_values = {}
    # Keep track of f_values for each threshold
    for i in range(255):  # should equal n_bins in metrics/f_measure.py
        f_values[i] = AverageMeter()

    if metrics_res == {}:
        metrics_res["f_scores"] = AverageMeter()
        metrics_res["f_maxs"] = AverageMeter()
        metrics_res["f_maxs_fixed"] = AverageMeter()
        metrics_res["f_means"] = AverageMeter()
        metrics_res["maes"] = AverageMeter()
        metrics_res["ious"] = AverageMeter()
        metrics_res["pixel_accs"] = AverageMeter()
        metrics_res["s_measures"] = AverageMeter()

    if reset:
        metrics_res["f_scores"].reset()
        metrics_res["f_maxs"].reset()
        metrics_res["f_maxs_fixed"].reset()
        metrics_res["f_means"].reset()
        metrics_res["maes"].reset()
        metrics_res["ious"].reset()
        metrics_res["pixel_accs"].reset()
        metrics_res["s_measures"].reset()

    # iterate over batch dimension
    for _, (pred_mask, gt_mask) in enumerate(zip(batch_pred_masks, batch_gt_masks)):
        assert pred_mask.shape == gt_mask.shape, f"{pred_mask.shape} != {gt_mask.shape}"
        assert len(pred_mask.shape) == len(gt_mask.shape) == 2
        # Compute
        # Binarize at 0.5 for IoU and pixel accuracy
        binary_pred = (pred_mask > 0.5).float().squeeze()
        iou = compute_iou(binary_pred, gt_mask)
        f_measures = FMeasure()(pred_mask, gt_mask)  # soft mask for F measure
        mae = compute_mae(binary_pred, gt_mask)
        pixel_acc = compute_pixel_accuracy(binary_pred, gt_mask)

        # Update
        metrics_res["ious"].update(val=iou.numpy(), n=1)
        metrics_res["f_scores"].update(val=f_measures["f_measure"].numpy(), n=1)
        metrics_res["f_maxs"].update(val=f_measures["f_max"].numpy(), n=1)
        metrics_res["f_means"].update(val=f_measures["f_mean"].numpy(), n=1)
        metrics_res["s_measures"].update(
            val=SMeasure()(pred_mask=pred_mask, gt_mask=gt_mask.to(torch.float32)), n=1
        )
        metrics_res["maes"].update(val=mae.numpy(), n=1)
        metrics_res["pixel_accs"].update(val=pixel_acc.numpy(), n=1)

        # Keep track of f_values for each threshold
        all_f = f_measures["all_f"].numpy()
        for k, v in f_values.items():
            v.update(val=all_f[k], n=1)
        # Then compute the max for the f_max_fixed
        metrics_res["f_maxs_fixed"].update(
            val=np.max([v.avg for v in f_values.values()]), n=1
        )

    results = {}
    # F-measure, F-max, F-mean, MAE, S-measure, IoU, pixel acc.
    results["f_measure"] = metrics_res["f_scores"].avg
    results["f_max"] = metrics_res["f_maxs"].avg
    results["f_maxs_fixed"] = metrics_res["f_maxs_fixed"].avg
    results["f_mean"] = metrics_res["f_means"].avg
    results["s_measure"] = metrics_res["s_measures"].avg
    results["mae"] = metrics_res["maes"].avg
    results["iou"] = float(iou.numpy())
    results["pixel_acc"] = metrics_res["pixel_accs"].avg

    return results, metrics_res


def evaluate_saliency(
    dataset,
    model,
    writer=None,
    batch_size=1,
    n_iter=-1,
    apply_bilateral=False,
    im_fullsize=True,
    method="pred",  # can also be "bkg",
    apply_weights: bool = True,
    evaluation_mode: str = "single",  # choices are ["single", "multi"]
):

    if im_fullsize:
        # Change transformation
        dataset.fullimg_mode()
        batch_size = 1

    valloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    sigmoid = nn.Sigmoid()

    metrics_res = {}
    metrics_res_bs = {}
    valbar = tqdm(enumerate(valloader, 0), leave=None)
    for i, data in valbar:
        inputs, _, _, _, _, gt_labels, _ = data
        inputs = inputs.to("cuda")
        gt_labels = gt_labels.to("cuda").float()

        # Forward step
        with torch.no_grad():
            preds = model(inputs, for_eval=True)

        h, w = gt_labels.shape[-2:]
        preds_up = F.interpolate(
            preds,
            scale_factor=model.vit_patch_size,
            mode="bilinear",
            align_corners=False,
        )[..., :h, :w]
        soft_preds = sigmoid(preds_up.detach()).squeeze(0)
        preds_up = (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()

        reset = True if i == 0 else False
        if evaluation_mode == "single":
            labeled, nr_objects = ndimage.label(preds_up.squeeze().cpu().numpy())
            if nr_objects == 0:
                preds_up_one_cc = preds_up.squeeze()
                print("nr_objects == 0")
            else:
                nb_pixel = [np.sum(labeled == i) for i in range(nr_objects + 1)]
                pixel_order = np.argsort(nb_pixel)

                cc = [torch.Tensor(labeled == i) for i in pixel_order]
                cc = torch.stack(cc).cuda()

                # Find CC set as background, here not necessarily the biggest
                cc_background = (
                    (
                        (
                            (~(preds_up[None, :, :, :].bool())).float()
                            + cc[:, None, :, :].cuda()
                        )
                        > 1
                    )
                    .sum(-1)
                    .sum(-1)
                    .argmax()
                )
                pixel_order = np.delete(pixel_order, int(cc_background.cpu().numpy()))

                preds_up_one_cc = torch.Tensor(labeled == pixel_order[-1]).cuda()

            _, metrics_res = eval_batch(
                gt_labels,
                preds_up_one_cc.unsqueeze(0),
                metrics_res=metrics_res,
                reset=reset,
            )

        elif evaluation_mode == "multi":
            # Eval without bilateral solver
            _, metrics_res = eval_batch(
                gt_labels,
                soft_preds.unsqueeze(0) if len(soft_preds.shape) == 2 else soft_preds,
                metrics_res=metrics_res,
                reset=reset,
            )  # soft preds needed for F beta measure

        # Apply bilateral solver
        preds_bs = None
        if apply_bilateral:
            get_all_cc = True if evaluation_mode == "multi" else False
            preds_bs, _ = batch_apply_bilateral_solver(
                data, preds_up.detach(), get_all_cc=get_all_cc
            )

            _, metrics_res_bs = eval_batch(
                gt_labels,
                preds_bs[None, :, :].float(),
                metrics_res=metrics_res_bs,
                reset=reset,
            )

        bar_str = (
            f"{dataset.name} | {evaluation_mode} mode | "
            f"F-max {metrics_res['f_maxs'].avg:.3f} "
            f"IoU {metrics_res['ious'].avg:.3f}, "
            f"PA {metrics_res['pixel_accs'].avg:.3f}"
        )

        if apply_bilateral:
            bar_str += (
                f" | with bilateral solver: "
                f"F-max {metrics_res_bs['f_maxs'].avg:.3f}, "
                f"IoU {metrics_res_bs['ious'].avg:.3f}, "
                f"PA. {metrics_res_bs['pixel_accs'].avg:.3f}"
            )

        valbar.set_description(bar_str)

    # Writing in tensorboard
    if writer is not None:
        write_metric_tf(
            writer,
            metrics_res,
            n_iter=n_iter,
            name=f"{dataset.name}_{evaluation_mode}_",
        )

        if apply_bilateral:
            write_metric_tf(
                writer,
                metrics_res_bs,
                n_iter=n_iter,
                name=f"{dataset.name}_{evaluation_mode}-BS_",
            )

    # Go back to original transformation
    if im_fullsize:
        dataset.training_mode()

@torch.no_grad()
def eval_batch_student(batch_gt_masks, batch_pred_masks, metrics_res={}, reset=False):
    """
    Evaluates a batch of predictions for the student model without debugging prints.
    """
    f_values = {i: AverageMeter() for i in range(255)}  # F-measure thresholds

    # Initialize metrics_res if empty
    if not metrics_res:
        metrics_res = {
            "f_scores": AverageMeter(),
            "f_maxs": AverageMeter(),
            "f_maxs_fixed": AverageMeter(),
            "f_means": AverageMeter(),
            "maes": AverageMeter(),
            "ious": AverageMeter(),
            "pixel_accs": AverageMeter(),
            "s_measures": AverageMeter(),
        }

    # Reset metrics if specified
    if reset:
        for meter in metrics_res.values():
            meter.reset()

    # Iterate over batch of predictions and ground truth masks
    for pred_mask, gt_mask in zip(batch_pred_masks, batch_gt_masks):
        pred_mask = pred_mask.squeeze()
        gt_mask = gt_mask.squeeze()

        # Binarize the prediction mask at 0.5 threshold
        binary_pred = (pred_mask > 0.5).float()

        # Calculate metrics
        iou = compute_iou(binary_pred, gt_mask)
        f_measures = FMeasure()(pred_mask, gt_mask)
        mae = compute_mae(binary_pred, gt_mask)
        pixel_acc = compute_pixel_accuracy(binary_pred, gt_mask)

        # Update metrics
        metrics_res["ious"].update(val=iou.numpy(), n=1)
        metrics_res["f_scores"].update(val=f_measures["f_measure"].numpy(), n=1)
        metrics_res["f_maxs"].update(val=f_measures["f_max"].numpy(), n=1)
        metrics_res["f_means"].update(val=f_measures["f_mean"].numpy(), n=1)
        metrics_res["s_measures"].update(SMeasure()(pred_mask, gt_mask), n=1)
        metrics_res["maes"].update(val=mae.numpy(), n=1)
        metrics_res["pixel_accs"].update(val=pixel_acc.numpy(), n=1)

        # Track F-measure at different thresholds
        all_f = f_measures["all_f"].numpy()
        for k, v in f_values.items():
            v.update(val=all_f[k], n=1)

        # Update f_maxs_fixed with max of f_values at different thresholds
        metrics_res["f_maxs_fixed"].update(val=np.max([v.avg for v in f_values.values()]), n=1)

    # Final average metrics
    results = {k: v.avg for k, v in metrics_res.items()}
    return results, metrics_res


@torch.no_grad()
def student_evaluation_saliency(
        dataset,
        student_model,
        batch_size=1,
        apply_bilateral=False,
        im_fullsize=True,
        evaluation_mode="multi",
        output_dir="outputs/evaluation"
):
    """
    Evaluates the StudentModel for saliency detection on a specified dataset.

    Parameters:
    - dataset: Dataset to evaluate on.
    - student_model: The student model to be evaluated.
    - batch_size: Number of images per batch.
    - apply_bilateral: Whether to apply a bilateral solver for smoothing.
    - im_fullsize: Whether to evaluate at full image size.
    - evaluation_mode: Mode of evaluation ("single" or "multi").
    - output_dir: Directory to save the output images.
    """
    student_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)

    # Create output subfolders
    os.makedirs(os.path.join(output_dir, "predicted"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)

    if im_fullsize:
        dataset.fullimg_mode()
        batch_size = 1

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    sigmoid = nn.Sigmoid()
    metrics_res = {}

    valbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    for i, data in valbar:
        inputs = data[0]  # Adjusted to directly access inputs
        gt_labels = data[5].to(device).float()

        inputs = inputs.to(device)  # Move inputs to the appropriate device
        # Generate predictions
        preds = student_model(inputs)

        # Resize predictions to match ground truth dimensions
        h, w = gt_labels.shape[-2:]
        preds_up = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
        soft_preds = sigmoid(preds_up).squeeze(0)  # Soft prediction for F-measure

        binary_preds = (soft_preds > 0.5).float()  # Binary prediction for IoU and Pixel Accuracy

        # Save every 10th image
        if i % 10 == 0:
            save_image(inputs.cpu().squeeze(0), f"{output_dir}/original/image_{i}.png")
            save_image(gt_labels.cpu().squeeze(0), f"{output_dir}/ground_truth/image_{i}.png")
            binary_preds_uint8 = (binary_preds.squeeze(0).cpu() > 0.5).float()  # Scale binary predictions to [0, 1]
            # Add channel dimension and save as expected by save_image
            save_image(binary_preds_uint8.unsqueeze(0), f"{output_dir}/predicted/image_{i}.png")

        reset = i == 0
        if evaluation_mode == "single":
            labeled, num_objects = ndimage.label(binary_preds.cpu().numpy())
            if num_objects == 0:
                preds_up_one_cc = binary_preds
            else:
                sizes = [np.sum(labeled == j) for j in range(1, num_objects + 1)]
                largest_cc = (labeled == (np.argmax(sizes) + 1))
                preds_up_one_cc = torch.tensor(largest_cc, dtype=torch.float32, device=device)

            _, metrics_res = eval_batch_student(
                gt_labels, preds_up_one_cc.unsqueeze(0), metrics_res=metrics_res, reset=reset
            )

        elif evaluation_mode == "multi":
            _, metrics_res = eval_batch_student(
                gt_labels, soft_preds.unsqueeze(0), metrics_res=metrics_res, reset=reset
            )

        # Bilateral solver option
        if apply_bilateral:
            preds_bs, _ = batch_apply_bilateral_solver(data, binary_preds.detach(),
                                                       get_all_cc=(evaluation_mode == "multi"))
            _, metrics_res = eval_batch_student(gt_labels, preds_bs[None, :, :].float(), metrics_res=metrics_res,
                                                reset=reset)

        # Update progress bar
        valbar.set_postfix(
            f_max=metrics_res.get("f_maxs", AverageMeter()).avg,
            IoU=metrics_res.get("ious", AverageMeter()).avg,
            pixel_acc=metrics_res.get("pixel_accs", AverageMeter()).avg,
        )

    return metrics_res