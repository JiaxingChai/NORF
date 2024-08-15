import torch


def dice_loss(pred, gt, sigmoid):
    if sigmoid:
        pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    gt = gt.contiguous().view(-1)
    inter = (pred * gt).sum()
    dice = (2. * inter + 1) / (pred.sum() + gt.sum() + 1)
    return 1 - dice


def dice_bce_loss(pred, gt, sigmoid=True):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, gt)
    if sigmoid:
        pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    gt = gt.contiguous().view(-1)
    inter = (pred * gt).sum()
    dice = (2. * inter + 1) / (pred.sum() + gt.sum() + 1)
    return 1 - dice + bce