import torch

def iou(outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection) / (union + 1e-06)  # We smooth our devision to avoid 0/0

    return iou  # Or thresholded.mean() if you are interested in average across the batch