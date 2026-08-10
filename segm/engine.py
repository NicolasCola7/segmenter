import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu

import gc

 # 1. Helper function for O(1) memory confusion matrix
# Notice: 'a < n' automatically ignores the Cityscapes IGNORE_LABEL (255)
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        with amp_autocast():
            seg_pred = model.forward(im)
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    return logger


import gc
import numpy as np
import torch

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
   

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    # 2. Setup the running metrics matrix
    n_cls = data_loader.unwrapped.n_cls
    hist = np.zeros((n_cls, n_cls))

    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        
        # 3. Accumulate metrics immediately (bypassing the memory-heavy dict)
        gt_mask = val_seg_gt[filename]
        hist += fast_hist(gt_mask.flatten(), seg_pred.flatten(), n_cls)

        # 4. Force garbage collection to keep RAM perfectly flat
        del seg_pred
        del ims
        del batch
        gc.collect()

    # 5. Distributed gathering (Syncs a tiny 19x19 matrix instead of writing GBs to disk)
    if ptu.distributed:
        torch.distributed.barrier()
        hist_tensor = torch.tensor(hist, device=ptu.device)
        torch.distributed.all_reduce(hist_tensor)
        hist = hist_tensor.cpu().numpy()

    # 6. Compute final metrics exactly as `compute_metrics()` would
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-8)
    acc = np.diag(hist).sum() / (hist.sum() + 1e-8)
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-8)
    
    scores = {
        "aAcc": acc,
        "mAcc": np.nanmean(acc_cls),
        "mIoU": np.nanmean(iu),
    }

    # 7. Log stats exactly as before
    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger