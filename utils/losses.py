import torch.nn.functional as F


# Build Losses
def get_normal_loss(prediction, gt, ignore_label):
    '''Compute normal loss. (normalized cosine distance)
    Args:
      prediction: the output of cnn. Float type
      gt: the groundtruth. Float type
    '''
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    mask = ((gt == ignore_label).sum(dim=1) - 3).nonzero().squeeze()
    prediction = prediction[mask]
    gt = gt[mask]
    loss = F.cosine_similarity(gt, prediction)
    return 1 - loss.mean()
