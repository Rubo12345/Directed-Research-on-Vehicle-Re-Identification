from __future__ import absolute_import
from __future__ import division
from queue import Empty

import torch
import torch.nn as nn
import torch.nn.functional as F
from apply_2d_rotation import apply_2d_rotation

# Smoothed Cross Entropy Loss
class label_smooth_loss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * pred, dim=1).mean()

# Cross Entropy Loss
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing, if False, epsilon = 0
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

# Equivariance Constraint Loss
class EquivarianceConstraintLoss(nn.Module):
    def __init__(self, mode='l2', use_gpu=True):
        super(EquivarianceConstraintLoss, self).__init__()
        self.mode = mode
        self.use_gpu = use_gpu

    def forward(self, hp, hp_rot, label_rot):
        loss_l2 = 0.
        # loss_l1 = 0.
        loss_kl = 0.
        for r in range(4):
            mask = label_rot == r
            hp_masked = hp[mask].contiguous()
            hp_masked = apply_2d_rotation(hp_masked, rotation=r * 90)

            loss_l2 += torch.pow(hp_masked - hp_rot[mask], 2).sum()
            # loss_l1 += torch.abs(hp_masked - hp_rot[mask]).sum()
            loss_kl += (hp_masked * torch.log(hp_masked / hp_rot[mask].clamp(min=1e-9))).sum()

        loss_kl = loss_kl / hp.size(0)
        loss_l2 = loss_l2 / hp.nelement()
        # loss_l1 = loss_l1 / hp.nelement()
        return loss_kl * 0.4 + loss_l2 * 0.6

# Triplet Loss
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm(inputs,inputs.t(),beta =1,alpha=-2)   # My change
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i]].min().unsqueeze(0))
            # if dist[i][mask[i] == 0] != :
            #     dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0)) #Make sure to check this
            # else:
            #     dist_an.append(dist[i][mask[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# Triplet Loss 
class TripletLossSscl(nn.Module):
    """Triplet loss with hard positive/negative mining for sscl.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise cosine similarity, replace by the official when merged
        inputs = nn.functional.normalize(inputs, dim=1)
        sim = torch.einsum('nc,ck->nk', [inputs, inputs.T])
        sim = sim.clamp(min=1e-12)  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        sim_ap, sim_an = [], []
        for i in range(n):
            sim_ap.append(sim[i][mask[i]].min().unsqueeze(0))
            sim_an.append(sim[i][mask[i] == 0].max().unsqueeze(0))
        sim_ap = torch.cat(sim_ap)
        sim_an = torch.cat(sim_an)

        # Compute ranking hinge loss
        y = torch.ones_like(sim_an)
        # loss = self.ranking_loss(dist_an, dist_ap, y)
        loss = self.ranking_loss(sim_ap, sim_an, y)
        return loss

# Circle Loss
class CircleLoss(object):
    def __init__(self, s=64, m=0.35):
        self.m = m
        self.s = s

    def __call__(self, global_features, targets):
        global_features = normalize(global_features, axis=-1)

        sim_mat = torch.matmul(global_features, global_features.t())

        N = sim_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float() - torch.eye(N).to(sim_mat.device)
        is_pos = is_pos.bool()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        s_p = sim_mat[is_pos].contiguous().view(N, -1)
        s_n = sim_mat[is_neg].contiguous().view(N, -1)

        alpha_p = F.relu(-s_p.detach() + 1 + self.m)
        alpha_n = F.relu(s_n.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.s * alpha_p * (s_p - delta_p)
        logit_n = self.s * alpha_n * (s_n - delta_n)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

# Distill Loss
class DistillLoss(nn.Module):
    def __init__(self, t=16):
        super().__init__()
        self.t = t

    def forward(self, y_s, y_t):
        """
        :param y_s: student logits
        :param y_t: teacher logits
        :return:
        """
        p_s = F.log_softmax(y_s / self.t, dim=1)
        p_t = F.softmax(y_t / self.t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.t ** 2) / y_s.shape[0]
        return loss

# PKT Loss
class PKTLoss(nn.Module):
    def __init__(self, eps=0.0000001):
        super().__init__()
        self.eps = eps

    def forward(self, output_net, target_net):
        """
        :param output_net: student feature vector
        :param target_net: teacher feature vector
        :return:
        """
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + self.eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + self.eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        # loss = torch.mean(target_similarity * torch.log((target_similarity + self.eps) / (model_similarity + self.eps)))
        loss = (target_similarity * torch.log((target_similarity + self.eps) / (model_similarity + self.eps))).sum(dim=1).mean()
        return loss

# Medium Distill MSE Loss
class MediumDistillMSELoss(nn.Module):
    def __init__(self, EPS=1e-5):
        super().__init__()
        self.EPS = EPS
        self.softmax = nn.Softmax(dim=1)

    def forward(self, domain_norms_t, domain_norms_s, feats_t, feats_s, weights=[1.0, 1.0, 1.0, 0.0]):
        """
        :param domain_norms: instance normalization layers
        :param feats_t: teacher features of layer 1, 2, 3, 4
        :param feats_s: student features
        :return:
        """
        assert len(set((len(domain_norms_t), len(feats_t), len(feats_s), len(weights)))) == 1
        num_layers = len(feats_t)

        # instance norm + l1
        feats_t_norm = []
        for i in range(num_layers):
            feats_t_norm.append(domain_norms_t[i](feats_t[i].detach()))
        # conv1x1, batch norm
        feats_s_norm = []
        for i in range(num_layers):
            feats_s_norm.append(domain_norms_s[i](feats_s[i]))

        loss = torch.tensor(0.0)
        for i in range(num_layers):
            # loss = loss + torch.pow((feats_t_norm[i] - feats_s[i]), 2).mean() * weights[i]
            loss = torch.abs(feats_t_norm[i].mean(dim=(2,3)) - feats_s_norm[i].mean(dim=(2,3))).mean() * weights[i] + loss
            # loss = loss + torch.abs(feats_t_norm[i] - feats_s[i]).mean() * weights[i]
        return loss

def triplet_loss(margin):
    loss = TripletLoss(
        margin=margin
    )   
    return loss



