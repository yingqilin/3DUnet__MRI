import torch


def F1(logit, target):
  """
  output: tensor of size (n_batch, image.shape)
  target: LongTensor of shape (n_batch, n_class, image.shape)
  """
  epsilon = 1e-20

  TP = ((logit == 1) * (target == 1)).sum()
  predicted_P = (logit == 1).sum()
  actual_P = (target == 1).sum()

  precision = (TP + epsilon) / (predicted_P + epsilon)
  recall = (TP + epsilon) / (actual_P + epsilon)
  f1_score = 2 * (precision * recall) / (precision + recall)

  return precision, recall, f1_score


def IOU(output, target):
  """
  output: tensor of size (n_batch, n_class, image.shape)
  target: LongTensor of shape (n_batch, n_class, image.shape)
  """
  epsilon = 1e-20

  dims = (0, *range(2, len(output.shape)))
  target = torch.zeros_like(output).scatter_(1, target[:, None, :], 1)
  intersection = output * target
  union = output + target - intersection
  iou = (intersection.sum(dim=dims).float() + epsilon) / (union.sum(dim=dims) + epsilon)

  return iou