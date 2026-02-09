import torch.nn as nn
import torch.nn.functional as F

from .softmax_loss_msi import CrossEntropyLabelSmooth
from .triplet_loss_msi import TripletLoss, cosine_dist


class ReIDLoss(nn.Module):
    """Build the loss function for ReID tasks. 
    """
    def __init__(self, args, num_classes):
        super(ReIDLoss, self).__init__()

        self.triplet = TripletLoss(args.MARGIN, normalize_feature=True)
        self.softmax = CrossEntropyLabelSmooth(num_classes=num_classes)

    def forward(self, feats, logits, target):
        triplet_loss = self.triplet(feats, target)[0]
        softmax_loss = self.softmax(logits, target)
        return softmax_loss + triplet_loss


def build_criterion(args, num_classes):
    return ReIDLoss(args, num_classes)
