import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True) #最远（难）的正样本
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True) #最近（难）的负样本
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False): #不对feature进行normalize
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)  # batchsize x 2048
        dist_mat = euclidean_dist(global_feat, global_feat)  # batchsize x batchsize #样本两两间的欧式距离
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)  # batchsize x 1 batchsize x 1 #每个样本作为anchor时，最难正样本、最难负样本距离

        # 根据hard_factor再调节一下anchor与最难正负样本间的距离
        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1) #向MarginRankingLoss第三个参数传入1，希望下面dist_an>dist_ap
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an






def PN_distance_splitting(dist_mat, labels):

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap = dist_mat[is_pos].contiguous().view(N, -1) # 正样本
    # print(dist_ap.shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an = dist_mat[is_neg].contiguous().view(N, -1) # 负样本
    # print(dist_an.shape)

    return dist_ap, dist_an

def weight_computing(dist_ap, dist_an):

    assert len(dist_ap.size()) == 2
    assert dist_ap.size(0) == dist_an.size(0)
    N = dist_ap.size(0)
    assert dist_ap.size(1) + dist_an.size(1) == N

    e_dist_ap = torch.exp(dist_ap).sum(dim=1,keepdim=True)
    e_dist_an = torch.exp(-dist_an).sum(dim=1,keepdim=True)
    wp = torch.exp(dist_ap)/e_dist_ap
    wn = torch.exp(-dist_an)/e_dist_an

    return wp, wn

class BatchSoftTripletLoss(object):

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False): #不对feature进行normalize
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)  # batchsize x 2048
        dist_mat = euclidean_dist(global_feat, global_feat)  # batchsize x batchsize #样本两两间的欧式距离
        dist_ap, dist_an = PN_distance_splitting(dist_mat, labels)  # batchsize x (P-1) batchsize x (batchsize-(P-1)) #每个样本作为anchor时，与所有正样本、负样本距离
        w_ap, w_an = weight_computing(dist_ap, dist_an)  # batchsize x (P-1) batchsize x (batchsize-(P-1)) #每个样本作为anchor时，所有正样本、负样本的权重
        # with torch.no_grad():
        #   w_ap, w_an = weight_computing(dist_ap, dist_an)  # batchsize x (P-1) batchsize x (batchsize-(P-1)) #每个样本作为anchor时，所有正样本、负样本的权重

        # # 根据hard_factor再调节一下anchor与最难正负样本间的距离
        # dist_ap *= (1.0 + self.hard_factor)
        # dist_an *= (1.0 - self.hard_factor)

        loss = (torch.log(1+torch.exp((w_ap*dist_ap).sum(dim=1)-(w_an*dist_an).sum(dim=1)))).mean()

        return loss, dist_ap, dist_an
